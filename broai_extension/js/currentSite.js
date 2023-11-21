// currentSite.js
function getCurrentTabInfo() {
    chrome.tabs.query({ active: true, currentWindow: true }, function (tabs) {
        const currentTab = tabs[0];
        displayCurrentTabInfo(currentTab);
    });
}

function displayCurrentTabInfo(tab) {
    const currentSiteSection = document.getElementById('currentSiteSection');
    currentSiteSection.innerHTML = `
        <div>URL: ${tab.url}</div>
        <div>Title: ${tab.title}</div>
    `;
}
