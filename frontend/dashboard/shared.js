/* ==========================================================================
   AI Risk Manager — Shared Component JavaScript
   Tooltip, mobile nav, disclaimer banner — no animation library.
   ========================================================================== */

'use strict';

/* ——————————————————————————————————————————————————————————————————————————
   TOOLTIP
   Accessible tooltip via aria-describedby.  Shows on hover (mouse) and
   on focus (keyboard).  CSS handles the opacity transition; JS sets up
   the ARIA wiring and repositioning when the tooltip would overflow the
   viewport.
   —————————————————————————————————————————————————————————————————————————— */

function initTooltips() {
  const tooltips = document.querySelectorAll('.tooltip-wrap');

  tooltips.forEach((wrap, idx) => {
    const trigger = wrap.querySelector('.tooltip-trigger');
    const content = wrap.querySelector('.tooltip-content');
    if (!trigger || !content) return;

    // Wire ARIA
    const id = `tooltip-${idx}`;
    content.id = id;
    content.setAttribute('role', 'tooltip');
    trigger.setAttribute('aria-describedby', id);

    // Ensure trigger is focusable
    if (!trigger.hasAttribute('tabindex') && trigger.tagName !== 'BUTTON' && trigger.tagName !== 'A') {
      trigger.setAttribute('tabindex', '0');
    }

    // Reposition if overflowing viewport
    function reposition() {
      content.style.left = '';
      content.style.transform = '';
      const rect = content.getBoundingClientRect();
      if (rect.left < 8) {
        content.style.left = '0';
        content.style.transform = 'none';
      } else if (rect.right > window.innerWidth - 8) {
        content.style.left = 'auto';
        content.style.right = '0';
        content.style.transform = 'none';
      }
    }

    wrap.addEventListener('mouseenter', reposition);
    trigger.addEventListener('focus', reposition);
  });
}


/* ——————————————————————————————————————————————————————————————————————————
   MOBILE NAV
   Toggle the mobile dropdown; close on Escape and outside click.
   —————————————————————————————————————————————————————————————————————————— */

function initMobileNav() {
  const btn = document.querySelector('.navbar__menu-btn');
  const menu = document.querySelector('.navbar__mobile-menu');
  if (!btn || !menu) return;

  function toggleMenu() {
    const isOpen = menu.classList.toggle('is-open');
    btn.setAttribute('aria-expanded', String(isOpen));
  }

  function closeMenu() {
    menu.classList.remove('is-open');
    btn.setAttribute('aria-expanded', 'false');
  }

  btn.addEventListener('click', toggleMenu);

  // Close on Escape
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && menu.classList.contains('is-open')) {
      closeMenu();
      btn.focus();
    }
  });

  // Close on outside click
  document.addEventListener('click', (e) => {
    if (menu.classList.contains('is-open') && !menu.contains(e.target) && !btn.contains(e.target)) {
      closeMenu();
    }
  });

  // Close when a link is clicked
  menu.querySelectorAll('.navbar__link').forEach(link => {
    link.addEventListener('click', closeMenu);
  });
}


/* ——————————————————————————————————————————————————————————————————————————
   INIT
   —————————————————————————————————————————————————————————————————————————— */

document.addEventListener('DOMContentLoaded', () => {
  initTooltips();
  initMobileNav();
});
