import * as THREE from 'https://esm.sh/three@0.160.0';
const container = document.getElementById('bg-canvas');
const vertexShader = `void main() { gl_Position = vec4(position, 1.0); }`;
const fragmentShader = `
  precision highp float; uniform vec2 resolution; uniform float time;
  void main(void) {
    vec2 uv = (gl_FragCoord.xy * 2.0 - resolution.xy) / min(resolution.x, resolution.y);
    float t = time * 0.05; vec3 color = vec3(0.0);
    for (int j = 0; j < 3; j++) {
      for (int i = 0; i < 5; i++) {
        color[j] += 0.002 * float(i * i) / abs(fract(t - 0.01 * float(j) + float(i) * 0.01) * 5.0 - length(uv) + mod(uv.x + uv.y, 0.2));
      }
    }
    float mono = (color.r + color.g + color.b) / 3.0;
    vec3 finalColor = mix(color, mono * vec3(0.1, 0.5, 0.3), 1.0);
    gl_FragColor = vec4(finalColor * 0.6, 1.0);
  }
`;
const scene = new THREE.Scene(); const camera = new THREE.Camera(); camera.position.z = 1;
const material = new THREE.ShaderMaterial({ uniforms: { time: { value: 1.0 }, resolution: { value: new THREE.Vector2() } }, vertexShader, fragmentShader });
scene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), material));
const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
renderer.setPixelRatio(window.devicePixelRatio); container.appendChild(renderer.domElement);
const onResize = () => { renderer.setSize(container.clientWidth, container.clientHeight); material.uniforms.resolution.value.x = renderer.domElement.width; material.uniforms.resolution.value.y = renderer.domElement.height; };
onResize(); window.addEventListener("resize", onResize);

// Pause animation when tab is hidden, inference is running, or the user said so
const BG_PREF_KEY = "draw2_bg_anim";
let rafId = null;
let inferencePaused = false;
let userOff = false;
try { userOff = localStorage.getItem(BG_PREF_KEY) === "off"; } catch {}
const animate = () => {
    rafId = requestAnimationFrame(animate);
    material.uniforms.time.value += 0.03;
    renderer.render(scene, camera);
};
function syncAnimation() {
    const shouldRun = !document.hidden && !inferencePaused && !userOff;
    if (shouldRun && !rafId) animate();
    if (!shouldRun && rafId) { cancelAnimationFrame(rafId); rafId = null; }
}
function applyUserPref() {
    container.style.display = userOff ? "none" : "";
    syncAnimation();
}
renderer.render(scene, camera); // paint one frame immediately either way
applyUserPref();
document.addEventListener("visibilitychange", syncAnimation);

// pipeline.js calls this around inference; it knows nothing about Three.js.
window.__bgAnim = {
    pause()  {
        inferencePaused = true;
        syncAnimation();
        material.uniforms.time.value = 1.0; // reset to frame 1, not a mid-motion freeze
        renderer.render(scene, camera);
    },
    resume() { inferencePaused = false; syncAnimation(); },
    isEnabled() { return !userOff; },
    setEnabled(on) {
        userOff = !on;
        try { localStorage.setItem(BG_PREF_KEY, on ? "on" : "off"); } catch {}
        applyUserPref();
    }
};