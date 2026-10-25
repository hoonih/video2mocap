// 조건 B·C 오프라인 렌더 씬 (브라우저 쪽).
// 실시간 루프(requestAnimationFrame)를 쓰지 않는다. 드라이버(render.mjs)가 프레임마다
// applyFrame() → renderFrame()을 호출해 수동으로 전진시킨다. 같은 입력이면 같은 픽셀이 나와야 한다.
import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { VRMLoaderPlugin, VRMUtils } from "@pixiv/three-vrm";

const state = { renderer: null, scene: null, camera: null, vrm: null, expressionNames: new Map() };

export async function init({ width, height, modelUrl, background, camera, lights, style, options = {} }) {
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  document.body.appendChild(canvas);

  const renderer = new THREE.WebGLRenderer({ canvas, antialias: options.antialias ?? true, preserveDrawingBuffer: true });
  renderer.setPixelRatio(1);
  renderer.setSize(width, height, false);
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.setClearColor(new THREE.Color().setRGB(...background.map((v) => v / 255), THREE.SRGBColorSpace), 1);

  const scene = new THREE.Scene();
  for (const light of lights) {
    const obj = light.type === "directional"
      ? new THREE.DirectionalLight(0xffffff, light.intensity)
      : new THREE.AmbientLight(0xffffff, light.intensity);
    if (light.position) obj.position.set(...light.position);
    scene.add(obj);
  }

  // near/far를 피사체 거리에 맞게 좁혀 깊이 정밀도를 높인다. 너무 넓으면(0.05~20m) 머리카락·얼굴 경계에서
  // 깊이가 거의 같은 두 면이 실행마다 다르게 이겨 같은 입력에서도 픽셀이 달라졌다 (결정론 위반).
  const cam = new THREE.PerspectiveCamera(camera.fov, width / height, camera.near ?? 0.05, camera.far ?? 20);
  cam.position.set(...camera.position);
  cam.lookAt(new THREE.Vector3(...camera.target));
  state.size = { width, height };

  const loader = new GLTFLoader();
  loader.register((parser) => new VRMLoaderPlugin(parser));
  const gltf = await loader.loadAsync(modelUrl);
  const vrm = gltf.userData.vrm;
  VRMUtils.removeUnnecessaryVertices(gltf.scene);
  VRMUtils.combineSkeletons(gltf.scene);
  VRMUtils.rotateVRM0(vrm); // VRM 0.x는 -Z를 바라본다. +Z(카메라 쪽)로 돌린다.
  vrm.scene.traverse((obj) => { obj.frustumCulled = false; });
  scene.add(vrm.scene);
  vrm.update(0);

  // camera.followBone이 있으면 그 본의 월드 위치 + offset을 기준으로 잡는다 (모델 키가 달라도 같은 구도).
  // camera.followEachFrame이면 매 프레임 자세를 적용한 뒤 다시 잡는다 (검증용 손 확대 뷰).
  state.cameraConfig = camera;
  state.camera = cam;
  state.vrm = vrm;
  if (camera.followBone) placeCamera();

  applyStyle(vrm, style || {});
  state.view = options.view || "model";
  if (state.view === "bones") setupBonesView(vrm, width, height, background);

  // ARKit 이름(browInnerUp) ↔ 모델 표정 이름(BrowInnerUp 등) 대소문자 무시 매핑
  for (const expr of vrm.expressionManager.expressions) {
    state.expressionNames.set(expr.expressionName.toLowerCase(), expr.expressionName);
  }

  Object.assign(state, { renderer, scene, camera: cam, vrm });
  return {
    expressions: vrm.expressionManager.expressions.map((e) => e.expressionName),
    humanBones: Object.keys(vrm.humanoid.humanBones),
    style: state.styleReport,
    view: state.view,
    faceFeatures: state.bones ? state.bones.features.map((f) => `${f.kind}:${f.side}`) : null,
  };
}

// 판독성을 위한 외형 조정 (움직임·표정 계수는 건드리지 않는다).
//  - browColor: 눈썹 재질 색을 이 sRGB 색으로 바꾼다 (VRoid 눈썹은 피부색에 가까워 작은 화면에서 안 보임)
//  - hideBangs: 이마 앞에서 눈썹 높이까지 내려오는 앞머리 가닥을 숨긴다
function applyStyle(vrm, style) {
  const report = { browMaterials: 0, hiddenHairPrimitives: 0 };
  vrm.scene.updateMatrixWorld(true);
  if (style.browColor) {
    const color = new THREE.Color().setRGB(...style.browColor.map((v) => v / 255), THREE.SRGBColorSpace);
    vrm.scene.traverse((obj) => {
      if (!obj.isMesh) return;
      for (const mat of Array.isArray(obj.material) ? obj.material : [obj.material]) {
        if (!/FaceBrow/.test(mat.name)) continue;
        if (mat.color) mat.color.copy(color);
        if (mat.shadeColorFactor) mat.shadeColorFactor.copy(color);
        report.browMaterials++;
      }
    });
  }
  if (style.hideBangs) {
    // 눈썹 영역(양 눈 바깥 + margin, 눈 위 browLow~browHigh 미터)을 앞에서 가리는 머리카락 가닥만 숨긴다.
    const head = new THREE.Vector3();
    const leftEye = new THREE.Vector3();
    const rightEye = new THREE.Vector3();
    vrm.humanoid.getNormalizedBoneNode("head").getWorldPosition(head);
    vrm.humanoid.getNormalizedBoneNode("leftEye").getWorldPosition(leftEye);
    vrm.humanoid.getNormalizedBoneNode("rightEye").getWorldPosition(rightEye);
    const { margin, browLow, browHigh, frontOffset } = style.hideBangs;
    const eyeY = (leftEye.y + rightEye.y) / 2;
    const region = {
      minX: Math.min(leftEye.x, rightEye.x) - margin, maxX: Math.max(leftEye.x, rightEye.x) + margin,
      minY: eyeY + browLow, maxY: eyeY + browHigh,
    };
    vrm.scene.traverse((obj) => {
      if (!obj.isMesh) return;
      const name = [].concat(obj.material)[0].name;
      if (!/_HAIR$/.test(name) || /HairBack/.test(name)) return;
      const pos = obj.geometry.attributes.position;
      const v = new THREE.Vector3();
      let covers = false;
      for (let i = 0; i < pos.count && !covers; i++) {
        v.fromBufferAttribute(pos, i).applyMatrix4(obj.matrixWorld);
        covers = v.z > head.z + frontOffset && v.x > region.minX && v.x < region.maxX && v.y > region.minY && v.y < region.maxY;
      }
      if (covers) { obj.visible = false; report.hiddenHairPrimitives++; }
    });
  }
  state.styleReport = report;
}

// 디버그: 얼굴 관련 메시의 경계 루프 요약 (재질, 정점 수, 중심)
export function debugFaceLoops() {
  const out = [];
  const v = new THREE.Vector3();
  state.vrm.scene.traverse((obj) => {
    if (!obj.isMesh) return;
    const name = [].concat(obj.material)[0].name;
    if (!/Face_00_SKIN|FaceBrow|FaceEyeline|FaceMouth/.test(name)) return;
    for (const loop of boundaryLoops(obj)) {
      const c = new THREE.Vector3();
      for (const i of loop.indices) { obj.getVertexPosition(i, v); c.add(obj.localToWorld(v.clone())); }
      c.divideScalar(loop.indices.length);
      out.push({ material: name, n: loop.indices.length, center: c.toArray().map((x) => +x.toFixed(4)) });
    }
  });
  return out;
}

// 휴지 자세(정규화 T-포즈)에서 각 휴머노이드 본의 월드 위치. 리타게팅 솔버가 뼈 방향 기준으로 쓴다.
export function dumpRestPose() {
  const { vrm } = state;
  vrm.humanoid.resetNormalizedPose();
  vrm.update(0);
  vrm.scene.updateMatrixWorld(true);
  const positions = {};
  const v = new THREE.Vector3();
  for (const bone of Object.keys(vrm.humanoid.humanBones)) {
    const node = vrm.humanoid.getNormalizedBoneNode(bone);
    if (node) positions[bone] = node.getWorldPosition(v).toArray();
  }
  return { metaVersion: vrm.meta.metaVersion, positions, faceAnchors: faceAnchors(vrm) };
}

// 메시의 경계 루프(삼각형 하나에만 속한 모서리를 이은 고리)를 찾는다. 위치가 같은 정점은 하나로 본다
// (glTF는 UV 이음매에서 정점을 복제하므로). 반환: [{ indices: [정점 번호...] }], 각 루프는 닫힌 고리 순서.
function boundaryLoops(obj) {
  const geo = obj.geometry;
  const pos = geo.attributes.position;
  const index = geo.index ? geo.index.array : [...Array(pos.count).keys()];
  const key = (i) => `${pos.getX(i).toFixed(5)},${pos.getY(i).toFixed(5)},${pos.getZ(i).toFixed(5)}`;
  const canon = new Map(); const rep = new Array(pos.count);
  for (let i = 0; i < pos.count; i++) { const k = key(i); if (!canon.has(k)) canon.set(k, i); rep[i] = canon.get(k); }
  const edges = new Map();
  for (let t = 0; t < index.length; t += 3) {
    for (let e = 0; e < 3; e++) {
      const a = rep[index[t + e]], b = rep[index[t + (e + 1) % 3]];
      const k = a < b ? `${a}_${b}` : `${b}_${a}`;
      edges.set(k, (edges.get(k) || 0) + 1);
    }
  }
  const next = new Map();
  for (const [k, n] of edges) {
    if (n !== 1) continue;
    const [a, b] = k.split("_").map(Number);
    if (!next.has(a)) next.set(a, []); if (!next.has(b)) next.set(b, []);
    next.get(a).push(b); next.get(b).push(a);
  }
  const seen = new Set(); const loops = [];
  for (const start of [...next.keys()].sort((x, y) => x - y)) {
    if (seen.has(start)) continue;
    const loop = [start]; seen.add(start); let prev = -1; let cur = start;
    for (;;) {
      const nb = next.get(cur).filter((v) => v !== prev && !seen.has(v)).sort((x, y) => x - y);
      if (!nb.length) break;
      prev = cur; cur = nb[0]; seen.add(cur); loop.push(cur);
    }
    if (loop.length >= 3) loops.push({ indices: loop });
  }
  return loops;
}

// 얼굴 접촉 위치 매핑(mocap/rig.py FaceContactIK)용 얼굴 기준점: 휴지 자세 메시 정점에서 찾는다.
//  eyeOuterLeft/Right: 흰자 메시의 좌우 끝, noseTip: 얼굴 정중선에서 가장 앞, chin: 정중선 앞쪽에서 가장 아래,
//  forehead: 정중선 앞쪽에서 가장 위, mouth: jawOpen 모프가 가장 크게 움직이는 정중선 정점들의 가장 위(입술선)
function faceAnchors(vrm) {
  const head = new THREE.Vector3();
  vrm.humanoid.getNormalizedBoneNode("head").getWorldPosition(head);
  const v = new THREE.Vector3();
  const collect = (pattern) => {
    const out = [];
    vrm.scene.traverse((obj) => {
      if (!obj.isMesh || !pattern.test([].concat(obj.material)[0].name)) return;
      const pos = obj.geometry.attributes.position;
      for (let i = 0; i < pos.count; i++) {
        if (obj.isSkinnedMesh) obj.getVertexPosition(i, v); else v.fromBufferAttribute(pos, i);
        out.push({ p: obj.localToWorld(v.clone()), obj, i });
      }
    });
    return out;
  };
  const MIDLINE = 0.004;
  const eyes = collect(/EyeWhite/);
  const skin = collect(/Face_00_SKIN/).filter((e) => Math.abs(e.p.x - head.x) < MIDLINE && e.p.z > head.z);
  const pick = (arr, key) => arr.reduce((a, b) => (key(b) > key(a) ? b : a)).p.toArray();
  // jawOpen 모프 변위가 큰 정중선 정점
  let mouth = null;
  const jaw = skin.filter((e) => e.obj.morphTargetDictionary && "jawOpen" in e.obj.morphTargetDictionary);
  if (jaw.length) {
    const disp = jaw.map((e) => {
      const attr = e.obj.geometry.morphAttributes.position[e.obj.morphTargetDictionary.jawOpen];
      return { e, d: new THREE.Vector3().fromBufferAttribute(attr, e.i).length() };
    });
    const maxD = Math.max(...disp.map((x) => x.d));
    const moving = disp.filter((x) => x.d > 0.3 * maxD).map((x) => x.e);
    if (moving.length) mouth = pick(moving, (e) => e.p.y);
  }
  return {
    eyeOuterLeft: pick(eyes, (e) => e.p.x), eyeOuterRight: pick(eyes, (e) => -e.p.x),
    noseTip: pick(skin, (e) => e.p.z), chin: pick(skin, (e) => -e.p.y), forehead: pick(skin, (e) => e.p.y),
    mouth,
  };
}

// frame: { bones: {boneName: [x,y,z,w] 정규화 본 로컬 쿼터니언}, blendshapes: {arkitName: 0..1} }
export function applyFrame(frame) {
  const { vrm } = state;
  vrm.humanoid.resetNormalizedPose();
  // 입력 쿼터니언은 VRM 1.0 정규화 규약(모델 정면 +Z, 모델 왼쪽 +X, 위 +Y)이다.
  // VRM 0.x 모델의 정규화 본은 X·Z축이 반대라서 (x, y, z, w) → (-x, y, -z, w)로 바꿔 넣는다 (실측 확인).
  const flip = vrm.meta.metaVersion === "0" ? -1 : 1;
  const unknownBones = [];
  for (const [bone, q] of Object.entries(frame.bones || {})) {
    const node = vrm.humanoid.getNormalizedBoneNode(bone);
    if (node) node.quaternion.set(flip * q[0], q[1], flip * q[2], q[3]);
    else unknownBones.push(bone);
  }
  if (unknownBones.length) throw new Error(`모델에 없는 본: ${unknownBones.join(", ")}`);
  for (const expr of vrm.expressionManager.expressions) vrm.expressionManager.setValue(expr.expressionName, 0);
  const missing = [];
  for (const [name, value] of Object.entries(frame.blendshapes || {})) {
    const target = state.expressionNames.get(name.toLowerCase());
    if (target) vrm.expressionManager.setValue(target, value);
    else missing.push(name);
  }
  // 고정 dt로만 갱신한다 (스프링본 등 시간 의존 요소의 결정론 확보). 정지 렌더는 0.
  vrm.update(frame.dt ?? 0);
  return missing;
}

function placeCamera() {
  const { camera: cfg } = { camera: state.cameraConfig };
  const anchor = new THREE.Vector3();
  state.vrm.scene.updateMatrixWorld(true);
  // 실제 메시를 움직이는 raw 본의 위치를 따른다 (정규화 본은 자세 적용 후 위치가 갱신되지 않을 수 있음)
  state.vrm.humanoid.getRawBoneNode(cfg.followBone).getWorldPosition(anchor);
  state.camera.position.copy(anchor).add(new THREE.Vector3(...cfg.position));
  state.camera.lookAt(anchor.clone().add(new THREE.Vector3(...cfg.target)));
}

// frame.viewOffset = [dx, dy]: 화면에서 그림 전체를 (dx, dy) 픽셀 옮긴다 (원근은 그대로).
// 조건 A에서 화자가 움직인 만큼 B·C를 옮기는 데 쓴다. 카메라 투영은 Python(mocap/model_render.py)과 같다.
function applyViewOffset(offset) {
  const { width, height } = state.size;
  if (offset) state.camera.setViewOffset(width, height, -offset[0], -offset[1], width, height);
  else state.camera.clearViewOffset();
}

export function renderFrame(frame) {
  if (state.cameraConfig.followBone && state.cameraConfig.followEachFrame) placeCamera();
  applyViewOffset(frame && frame.viewOffset);
  state.vrm.scene.updateMatrixWorld(true);
  state.camera.updateMatrixWorld(true);
  let png;
  if (state.view === "bones") {
    png = drawBones();
  } else {
    state.renderer.render(state.scene, state.camera);
    png = state.renderer.domElement.toDataURL("image/png");
  }
  return { png, joints: projectedJoints() };
}

// ---------------------------------------------------------------------------
// 조건 B: 같은 리그의 뼈만 (2D 캔버스에 선과 점). 카메라·자세·표정은 조건 C와 같다.
// ---------------------------------------------------------------------------
const BONE_COLOR = 235;
const LEFT_DIM = 0.75;            // 좌우 구분: 모델 왼쪽 부위를 어둡게 (기존 SkeletonRenderer와 같은 규칙)
const BODY_LINE = 4, BODY_JOINT = 6, FINGER_LINE = 3, FINGER_JOINT = 3;
const FACE_LINE = 2, BROW_LINE = 3, NOSE_DOT = 3, HEAD_LINE = 2;
const FINGERTIP_EXTENSION = 0.75;
const ALIGN_JOINTS = [
  "hips", "spine", "chest", "upperChest", "neck", "head",
  "leftUpperArm", "leftLowerArm", "leftHand", "rightUpperArm", "rightLowerArm", "rightHand",
  "leftIndexProximal", "leftLittleProximal", "rightIndexProximal", "rightLittleProximal",
  "leftThumbDistal", "rightThumbDistal", "leftMiddleDistal", "rightMiddleDistal",
];
const FINGERS = ["Thumb", "Index", "Middle", "Ring", "Little"];

function project(v) {
  const p = v.clone().project(state.camera);
  return [(p.x + 1) / 2 * state.size.width, (1 - p.y) / 2 * state.size.height];
}

function bonePos(name) {
  const node = state.vrm.humanoid.getRawBoneNode(name);
  return node ? node.getWorldPosition(new THREE.Vector3()) : null;
}

function projectedJoints() {
  const out = {};
  for (const name of ALIGN_JOINTS) { const p = bonePos(name); if (p) out[name] = project(p); }
  return out;
}

function setupBonesView(vrm, width, height, background) {
  vrm.scene.traverse((obj) => { if (obj.isMesh) obj.visible = false; });  // 메시는 숨기고 자세·스키닝만 갱신
  const canvas = document.createElement("canvas");
  canvas.width = width; canvas.height = height;
  state.bones = { canvas, ctx: canvas.getContext("2d"), background, features: [] };

  // 뼈 연결 (부모 → 자식). 손가락 끝은 끝마디를 한 마디 길이만큼 연장한 점.
  const segs = [["hips", "spine"], ["spine", "chest"], ["chest", "upperChest"], ["upperChest", "neck"]];
  for (const side of ["left", "right"]) {
    const S = side;
    segs.push(["upperChest", `${S}Shoulder`], [`${S}Shoulder`, `${S}UpperArm`], [`${S}UpperArm`, `${S}LowerArm`],
              [`${S}LowerArm`, `${S}Hand`], ["hips", `${S}UpperLeg`], [`${S}UpperLeg`, `${S}LowerLeg`],
              [`${S}LowerLeg`, `${S}Foot`], [`${S}Foot`, `${S}Toes`]);
  }
  state.bones.segments = segs.filter(([a, b]) => vrm.humanoid.getRawBoneNode(a) && vrm.humanoid.getRawBoneNode(b));

  // 얼굴 윤곽: 얼굴 피부 메시의 눈·입 구멍 경계, 눈썹 메시 외곽, 코끝 정점, 머리 윤곽 원
  const anchors = faceAnchors(vrm);
  const near = (c, p, tol) => c.distanceTo(new THREE.Vector3(...p)) < tol;
  const v = new THREE.Vector3();
  const loopCenter = (obj, loop) => {
    const c = new THREE.Vector3();
    for (const i of loop.indices) { obj.getVertexPosition(i, v); c.add(obj.localToWorld(v.clone())); }
    return c.divideScalar(loop.indices.length);
  };
  const eyeMid = new THREE.Vector3(...anchors.eyeOuterLeft).add(new THREE.Vector3(...anchors.eyeOuterRight)).multiplyScalar(0.5);
  vrm.scene.traverse((obj) => {
    if (!obj.isMesh) return;
    const name = [].concat(obj.material)[0].name;
    const isSkin = /Face_00_SKIN/.test(name), isBrow = /FaceBrow/.test(name);
    if (!isSkin && !isBrow) return;
    for (const loop of boundaryLoops(obj)) {
      const c = loopCenter(obj, loop);
      const side = c.x >= eyeMid.x ? "left" : "right";
      if (isBrow) state.bones.features.push({ obj, indices: loop.indices, kind: "brow", side });
      else if (anchors.mouth && near(c, anchors.mouth, 0.02)) state.bones.features.push({ obj, indices: loop.indices, kind: "mouth", side: "center" });
      else if (Math.abs(c.y - eyeMid.y) < 0.02 && c.z > eyeMid.z - 0.03 && Math.abs(c.x - eyeMid.x) < 0.07)
        state.bones.features.push({ obj, indices: loop.indices, kind: "eye", side });
    }
    if (isSkin && !state.bones.nose) {
      let best = -1, bestD = Infinity;
      const pos = obj.geometry.attributes.position; const nose = new THREE.Vector3(...anchors.noseTip);
      for (let i = 0; i < pos.count; i++) { obj.getVertexPosition(i, v); const d = obj.localToWorld(v.clone()).distanceTo(nose); if (d < bestD) { bestD = d; best = i; } }
      if (bestD < 0.003) state.bones.nose = { obj, index: best };
    }
  });
  // 머리 윤곽 원: 얼굴 양옆 폭의 절반 × 1.15, 중심은 귀 높이의 머리 중앙. 머리 본에 붙여 따라 움직인다.
  const head = vrm.humanoid.getRawBoneNode("head");
  const center = eyeMid.clone(); center.z = new THREE.Vector3(...anchors.eyeOuterLeft).z - 0.06; center.y -= 0.01;
  const halfWidth = Math.abs(anchors.eyeOuterLeft[0] - anchors.eyeOuterRight[0]) / 2 + 0.03;
  state.bones.headLocal = head.worldToLocal(center.clone());
  state.bones.headRadius = halfWidth * 1.15;
  state.bones.featureCount = state.bones.features.length;
}

// 눈썹 카드 외곽(닫힌 고리)을 좌우 끝에서 위·아래 두 변으로 나눠, 같은 가로 위치끼리 평균한 중심선.
function browCenterline(loop, samples = 8) {
  const n = loop.length;
  let iMin = 0, iMax = 0;
  loop.forEach((p, i) => { if (p[0] < loop[iMin][0]) iMin = i; if (p[0] > loop[iMax][0]) iMax = i; });
  const walk = (from, to, step) => { const out = []; for (let i = from; ; i = (i + step + n) % n) { out.push(loop[i]); if (i === to) break; } return out; };
  const edgeA = walk(iMin, iMax, 1), edgeB = walk(iMin, iMax, -1);
  const at = (edge, x) => {
    for (let i = 1; i < edge.length; i++) {
      const [x0, y0] = edge[i - 1], [x1, y1] = edge[i];
      if ((x - x0) * (x - x1) <= 0 && x1 !== x0) return y0 + (y1 - y0) * (x - x0) / (x1 - x0);
    }
    return edge[edge.length - 1][1];
  };
  const xa = loop[iMin][0], xb = loop[iMax][0];
  const out = [];
  // 카드 양 끝의 세로 변이 섞이지 않도록 가로 길이의 양쪽 8%는 빼고 샘플링한다
  const lo = xa + (xb - xa) * 0.08, hi = xb - (xb - xa) * 0.08;
  for (let k = 0; k <= samples; k++) { const x = lo + (hi - lo) * k / samples; out.push([x, (at(edgeA, x) + at(edgeB, x)) / 2]); }
  return out;
}

function color(side) {
  const k = side === "left" ? LEFT_DIM : 1.0;
  const c = Math.round(BONE_COLOR * k);
  return `rgb(${c},${c},${c})`;
}

function sideOf(name) { return name.startsWith("left") ? "left" : name.startsWith("right") ? "right" : "center"; }

function drawBones() {
  const { ctx, canvas, background } = state.bones;
  ctx.fillStyle = `rgb(${background[0]},${background[1]},${background[2]})`;
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.lineCap = "round"; ctx.lineJoin = "round";
  const line = (a, b, width, side) => {
    ctx.strokeStyle = color(side); ctx.lineWidth = width;
    ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke();
  };
  const dot = (p, r, side) => { ctx.fillStyle = color(side); ctx.beginPath(); ctx.arc(p[0], p[1], r, 0, Math.PI * 2); ctx.fill(); };

  // 머리 윤곽
  const head = state.vrm.humanoid.getRawBoneNode("head");
  const headCenter = head.localToWorld(state.bones.headLocal.clone());
  const hc = project(headCenter);
  const camDist = state.camera.position.distanceTo(headCenter);
  const focal = (state.size.height / 2) / Math.tan(THREE.MathUtils.degToRad(state.camera.fov) / 2);
  const hr = focal * state.bones.headRadius / camDist;

  // 몸통·팔·다리 (목은 머리 윤곽에서 멈춘다: 머리 본까지 그으면 입을 가로지른다)
  for (const [a, b] of state.bones.segments) {
    let pa = project(bonePos(a)), pb = project(bonePos(b));
    if (b === "neck") {
      const neckTop = project(bonePos("neck")); const dx = neckTop[0] - hc[0], dy = neckTop[1] - hc[1];
      const d = Math.hypot(dx, dy);
      pb = d > hr ? neckTop : [hc[0] + dx / Math.max(d, 1e-6) * hr, hc[1] + dy / Math.max(d, 1e-6) * hr];
    }
    line(pa, pb, BODY_LINE, sideOf(b));
  }
  for (const [, b] of state.bones.segments) if (b !== "neck") dot(project(bonePos(b)), BODY_JOINT, sideOf(b));

  ctx.strokeStyle = color("center"); ctx.lineWidth = HEAD_LINE;
  ctx.beginPath(); ctx.arc(hc[0], hc[1], hr, 0, Math.PI * 2); ctx.stroke();

  // 얼굴 (블렌드셰이프·스키닝이 적용된 정점 위치)
  const v = new THREE.Vector3();
  for (const f of state.bones.features) {
    const pts = f.indices.map((i) => { f.obj.getVertexPosition(i, v); return project(f.obj.localToWorld(v.clone())); });
    ctx.strokeStyle = color("center");
    if (f.kind === "brow") {
      // 눈썹은 텍스처를 입힌 직사각형 카드라 외곽선은 상자처럼 보인다 → 카드의 가로 중심선을 그린다
      const center = browCenterline(pts);
      ctx.lineWidth = BROW_LINE; ctx.beginPath(); ctx.moveTo(center[0][0], center[0][1]);
      for (const p of center.slice(1)) ctx.lineTo(p[0], p[1]);
      ctx.stroke();
      continue;
    }
    ctx.lineWidth = FACE_LINE;
    ctx.beginPath(); ctx.moveTo(pts[0][0], pts[0][1]);
    for (const p of pts.slice(1)) ctx.lineTo(p[0], p[1]);
    ctx.closePath(); ctx.stroke();
  }
  if (state.bones.nose) {
    const { obj, index } = state.bones.nose; obj.getVertexPosition(index, v);
    dot(project(obj.localToWorld(v.clone())), NOSE_DOT, "center");
  }

  // 손 (맨 위): 손목 → 손가락 마디 → 손끝
  for (const side of ["left", "right"]) {
    const wrist = bonePos(`${side}Hand`); if (!wrist) continue;
    for (const finger of FINGERS) {
      const names = finger === "Thumb" ? ["Metacarpal", "Proximal", "Distal"] : ["Proximal", "Intermediate", "Distal"];
      const chain = names.map((n) => bonePos(`${side}${finger}${n}`));
      if (chain.some((p) => !p)) continue;
      // 손끝: 끝마디 방향으로 마디 길이의 0.75배 (1배면 메시 손끝보다 약간 튀어나왔다)
      const tip = chain[2].clone().add(chain[2].clone().sub(chain[1]).multiplyScalar(FINGERTIP_EXTENSION));
      const pts = [wrist, ...chain, tip].map(project);
      for (let i = 1; i < pts.length; i++) line(pts[i - 1], pts[i], FINGER_LINE, side);
      for (const p of pts.slice(1)) dot(p, FINGER_JOINT, side);
    }
  }
  return canvas.toDataURL("image/png");
}
