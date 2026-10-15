document.getElementById('theme-toggle').addEventListener('click', () => {
  document.documentElement.classList.toggle('dark');
});

// The animation module is a deferred module, so it is not on window yet here:
// read the stored preference directly for the initial icon.
const bgToggle = document.getElementById('bg-toggle');
if (bgToggle) {
  const iconOn = document.getElementById('bg-icon-on');
  const iconOff = document.getElementById('bg-icon-off');
  const stored = (() => { try { return localStorage.getItem('draw2_bg_anim'); } catch { return null; } })();
  let on = stored !== 'off';
  const paint = () => {
    iconOn.hidden = !on;
    iconOff.hidden = on;
    bgToggle.classList.toggle('bg-on', on);
    bgToggle.classList.toggle('!bg-emerald-500', on);
    bgToggle.classList.toggle('hover:!bg-emerald-400', on);
    bgToggle.classList.toggle('!text-white', on);
  };
  paint();
  bgToggle.addEventListener('click', () => {
    on = !on;
    window.__bgAnim?.setEnabled(on);
    paint();
  });
}
const langToggle = document.getElementById('lang-toggle');
const langMenu = document.getElementById('lang-menu');
function closeLangMenu() { langMenu.hidden = true; }
function markActiveLang() {
  langMenu.querySelectorAll('.lang-option').forEach(opt => {
    opt.classList.toggle('bg-black/5', opt.dataset.lang === window.getLang());
    opt.classList.toggle('dark:bg-white/10', opt.dataset.lang === window.getLang());
  });
}
langToggle.addEventListener('click', (e) => {
  e.stopPropagation();
  markActiveLang();
  langMenu.hidden = !langMenu.hidden;
});
langMenu.addEventListener('click', (e) => {
  const opt = e.target.closest('.lang-option');
  if (!opt) return;
  window.setLang(opt.dataset.lang);
  closeLangMenu();
});
document.addEventListener('click', (e) => {
  if (!langMenu.hidden && !langMenu.contains(e.target) && e.target !== langToggle) closeLangMenu();
});
document.addEventListener('keydown', (e) => { if (e.key === 'Escape') closeLangMenu(); });