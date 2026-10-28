'use strict';
const chooser = document.getElementById('example-select');
if (chooser) chooser.addEventListener('change', () => {
  document.querySelectorAll('.example').forEach((panel, index) => { panel.hidden = String(index) !== chooser.value; });
});
const steps = Array.from(document.querySelectorAll('.walk-step'));
function activateStep(step) {
  steps.forEach(button => {
    const selected = button === step;
    button.setAttribute('aria-expanded', String(selected));
    document.getElementById(button.getAttribute('aria-controls')).hidden = !selected;
  });
}
steps.forEach((button, index) => {
  button.addEventListener('click', () => activateStep(button));
  button.addEventListener('keydown', event => {
    let next;
    if (event.key === 'ArrowRight' || event.key === 'ArrowDown') next = (index + 1) % steps.length;
    else if (event.key === 'ArrowLeft' || event.key === 'ArrowUp') next = (index - 1 + steps.length) % steps.length;
    else if (event.key === 'Home') next = 0;
    else if (event.key === 'End') next = steps.length - 1;
    else return;
    event.preventDefault(); steps[next].focus(); activateStep(steps[next]);
  });
});
const status = document.querySelector('.copy-status');
document.querySelectorAll('[data-copy]').forEach(button => button.addEventListener('click', async () => {
  const source = document.getElementById(button.dataset.copy);
  try {
    if (!navigator.clipboard) throw new Error('Clipboard unavailable');
    await navigator.clipboard.writeText(source.textContent);
    if (status) status.textContent = 'Copied. Review the prompt or commands before running them.';
    const label = button.textContent; button.textContent = 'Copied';
    setTimeout(() => { button.textContent = label; }, 1800);
  } catch (_) {
    const selection = window.getSelection(); const range = document.createRange();
    range.selectNodeContents(source); selection.removeAllRanges(); selection.addRange(range);
    if (status) status.textContent = 'Text selected. Use your device’s copy command.';
  }
}));
