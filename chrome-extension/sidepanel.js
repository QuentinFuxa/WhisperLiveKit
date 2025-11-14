console.log("sidepanel.js");

async function run() {
  const micPermission = await navigator.permissions.query({
    name: "microphone",
  });

  document.getElementById(
    "audioPermission"
  ).innerText = `MICROPHONE: ${micPermission.state}`;

  if (micPermission.state !== "granted") {
    chrome.tabs.create({ url: "requestPermissions.html" });
  }

  const intervalId = setInterval(async () => {
    const micPermission = await navigator.permissions.query({
      name: "microphone",
    });
    if (micPermission.state === "granted") {
      document.getElementById(
        "audioPermission"
      ).innerText = `MICROPHONE: ${micPermission.state}`;
      clearInterval(intervalId);
    }
  }, 100);
}

void run();
