/* upload.js - drag/drop + progress upload via XHR (fetch doesn't expose upload progress) */

let selectedFile = null;

document.addEventListener("DOMContentLoaded", () => {
  const dropzone = document.getElementById("dropzone");
  const fileInput = document.getElementById("file-input");
  const uploadBtn = document.getElementById("upload-btn");

  dropzone.addEventListener("click", () => fileInput.click());
  dropzone.addEventListener("dragover", (e) => { e.preventDefault(); dropzone.classList.add("dragover"); });
  dropzone.addEventListener("dragleave", () => dropzone.classList.remove("dragover"));
  dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
    if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
  });
  fileInput.addEventListener("change", (e) => {
    if (e.target.files.length) handleFile(e.target.files[0]);
  });

  uploadBtn.addEventListener("click", doUpload);
});

function handleFile(file) {
  const validExt = [".mp4", ".avi", ".mov"];
  const ext = "." + file.name.split(".").pop().toLowerCase();
  const alertBox = document.getElementById("alert-box");

  if (!validExt.includes(ext)) {
    showAlert(alertBox, "Unsupported file type. Please choose MP4, AVI, or MOV.");
    return;
  }
  if (file.size > 200 * 1024 * 1024) {
    showAlert(alertBox, "File exceeds 200MB limit.");
    return;
  }
  alertBox.style.display = "none";
  selectedFile = file;

  document.getElementById("file-info").style.display = "block";
  document.getElementById("file-name").textContent = file.name;
  document.getElementById("file-size").textContent = `${(file.size / 1024 / 1024).toFixed(1)} MB`;
  document.getElementById("upload-btn").disabled = false;
}

function doUpload() {
  if (!selectedFile) return;
  const btn = document.getElementById("upload-btn");
  const fill = document.getElementById("progress-fill");
  const alertBox = document.getElementById("alert-box");
  btn.disabled = true; btn.textContent = "Uploading...";

  const formData = new FormData();
  formData.append("file", selectedFile);

  const xhr = new XMLHttpRequest();
  xhr.open("POST", `${API_BASE}/videos/upload`);
  xhr.setRequestHeader("Authorization", `Bearer ${Auth.getToken()}`);

  xhr.upload.addEventListener("progress", (e) => {
    if (e.lengthComputable) {
      fill.style.width = `${Math.round((e.loaded / e.total) * 100)}%`;
    }
  });

  xhr.onload = async () => {
    if (xhr.status === 201) {
      const video = JSON.parse(xhr.responseText);
      btn.textContent = "Starting analysis...";
      try {
        await apiFetch(`/videos/${video.id}/analyze`, { method: "POST" });
        window.location.href = `result.html?id=${video.id}`;
      } catch (e) {
        showAlert(alertBox, "Uploaded, but failed to start analysis: " + e.message);
        btn.disabled = false; btn.textContent = "Upload & Start Analysis";
      }
    } else {
      let msg = "Upload failed";
      try { msg = JSON.parse(xhr.responseText).detail; } catch {}
      showAlert(alertBox, msg);
      btn.disabled = false; btn.textContent = "Upload & Start Analysis";
    }
  };
  xhr.send(formData);
}
