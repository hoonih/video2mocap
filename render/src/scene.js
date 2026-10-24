// 조건 B·C 오프라인 렌더 씬 (브라우저 쪽).
// 실시간 루프(requestAnimationFrame)를 쓰지 않는다. 드라이버(render.mjs)가 프레임마다
// applyFrame() → renderFrame()을 호출해 수동으로 전진시킨다. 같은 입력이면 같은 픽셀이 나와야 한다.
import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { VRMLoaderPlugin, VRMUtils } from "@pixiv/three-vrm";

const state = { renderer: null, scene: null, camera: null, vrm: null, expressionNames: new Map() };

export async function init({ width, height, modelUrl, background, camera, lights, style }) {
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  document.body.appendChild(canvas);

  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, preserveDrawingBuffer: true });
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

  const cam = new THREE.PerspectiveCamera(camera.fov, width / height, 0.05, 20);
  cam.position.set(...camera.position);
  cam.lookAt(new THREE.Vector3(...camera.target));

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

  // camera.followBone이 있으면 그 본의 월드 위치 + offset을 기준으로 잡는다 (모델 키가 달라도 같은 구도)
  if (camera.followBone) {
    const anchor = new THREE.Vector3();
    vrm.humanoid.getNormalizedBoneNode(camera.followBone).getWorldPosition(anchor);
    cam.position.copy(anchor).add(new THREE.Vector3(...camera.position));
    cam.lookAt(anchor.clone().add(new THREE.Vector3(...camera.target)));
  }

  applyStyle(vrm, style || {});

  // ARKit 이름(browInnerUp) ↔ 모델 표정 이름(BrowInnerUp 등) 대소문자 무시 매핑
  for (const expr of vrm.expressionManager.expressions) {
    state.expressionNames.set(expr.expressionName.toLowerCase(), expr.expressionName);
  }

  Object.assign(state, { renderer, scene, camera: cam, vrm });
  return {
    expressions: vrm.expressionManager.expressions.map((e) => e.expressionName),
    humanBones: Object.keys(vrm.humanoid.humanBones),
    style: state.styleReport,
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
    const head = new THREE.Vector3();
    vrm.humanoid.getNormalizedBoneNode("head").getWorldPosition(head);
    const brow = new THREE.Vector3();
    // 눈썹 높이: 왼눈 본 위 hideBangs.aboveEye 미터
    vrm.humanoid.getNormalizedBoneNode("leftEye").getWorldPosition(brow);
    const cutoffY = brow.y + style.hideBangs.aboveEye;
    vrm.scene.traverse((obj) => {
      if (!obj.isMesh || !/HAIR/.test([].concat(obj.material)[0].name) || /HairBack/.test([].concat(obj.material)[0].name)) return;
      obj.geometry.computeBoundingBox();
      const box = obj.geometry.boundingBox.clone().applyMatrix4(obj.matrixWorld);
      const inFront = box.max.z > head.z + style.hideBangs.frontOffset;
      const reachesBrow = box.min.y < cutoffY && box.max.y > cutoffY - 0.1;
      const central = Math.abs((box.min.x + box.max.x) / 2 - head.x) < style.hideBangs.halfWidth;
      if (inFront && reachesBrow && central) { obj.visible = false; report.hiddenHairPrimitives++; }
    });
  }
  state.styleReport = report;
}

// frame: { bones: {boneName: [x,y,z,w] 정규화 본 로컬 쿼터니언}, blendshapes: {arkitName: 0..1} }
export function applyFrame(frame) {
  const { vrm } = state;
  vrm.humanoid.resetNormalizedPose();
  for (const [bone, q] of Object.entries(frame.bones || {})) {
    const node = vrm.humanoid.getNormalizedBoneNode(bone);
    if (node) node.quaternion.set(q[0], q[1], q[2], q[3]);
  }
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

export function renderFrame() {
  state.renderer.render(state.scene, state.camera);
  return state.renderer.domElement.toDataURL("image/png");
}
