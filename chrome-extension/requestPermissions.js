/**
 * Requests user permission for microphone access.
 * @returns {Promise<void>} A Promise that resolves when permission is granted or rejects with an error.
 */
async function getUserPermission() {
  console.log("Getting user permission for microphone access...");
  await navigator.mediaDevices.getUserMedia({ audio: true });
  const micPermission = await navigator.permissions.query({
    name: "microphone",
  });
  if (micPermission.state == "granted") {
    window.close();
  }
}

// Call the function to request microphone permission
getUserPermission();
