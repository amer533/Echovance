// Minimal progressive enhancement: busy state + double-submit protection for forms.
//
// Important: a submit button's name/value (e.g. status=approved on the action page) is only sent if
// that button is still enabled when the browser builds the form data, which happens AFTER this
// handler runs. So we copy the clicked button's name/value into a hidden input before disabling
// the buttons. (Bug fixed: disabling the button directly dropped "status" -> "This field is required".)
document.addEventListener("submit", (event) => {
  const form = event.target;
  if (form.dataset.submitted) {
    event.preventDefault(); // second click while the first request is in flight
    return;
  }
  form.dataset.submitted = "1";
  const submitter = event.submitter || form.querySelector("button[type=submit]");
  if (submitter && submitter.name) {
    const mirror = document.createElement("input");
    mirror.type = "hidden";
    mirror.name = submitter.name;
    mirror.value = submitter.value;
    mirror.dataset.submitterMirror = "1";
    form.appendChild(mirror);
  }
  form.querySelectorAll("button[type=submit]").forEach((button) => {
    button.disabled = true;
  });
  if (submitter) {
    submitter.setAttribute("aria-busy", "true");
    if (submitter.dataset.busyText) submitter.textContent = submitter.dataset.busyText;
  }
});

// Pages restored from the back/forward cache must be usable again.
window.addEventListener("pageshow", (event) => {
  if (!event.persisted) return;
  document.querySelectorAll("form[data-submitted]").forEach((form) => {
    delete form.dataset.submitted;
    form.querySelectorAll("[data-submitter-mirror]").forEach((input) => input.remove());
    form.querySelectorAll("button[type=submit]").forEach((button) => {
      button.disabled = false;
      button.removeAttribute("aria-busy");
    });
  });
});
