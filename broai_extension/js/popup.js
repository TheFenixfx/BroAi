document.addEventListener('DOMContentLoaded', initialize)
 
let url, method, headers, requestBody, response, history = [], isFormattedView = false;

/* "/js/popup.js", to manifest as content script if needed*/ 
function initialize() {
  // Assign DOM elements to variables and set up event listeners
  url = document.getElementById('url');
  url.value = "http://127.0.0.1:5000/ask";
  const getCurrentSiteInfoButton = document.getElementById('getCurrentSiteInfoButton');
  const viewBookmarksButton = document.getElementById('viewBookmarksButton');

  getCurrentSiteInfoButton.addEventListener('click', getCurrentTabInfo);
  viewBookmarksButton.addEventListener('click', getUserBookmarks);
  method = document.getElementById('method');
  system = document.getElementById('system');
  requestBody = document.getElementById('requestBody');

  response = document.getElementById('response');  
  document.getElementById('toggleViewButton').addEventListener('click', toggleView);
  document.getElementById('exportButton').addEventListener('click', exportHistory);
  document.getElementById('importButton').addEventListener('click', importHistory);
  document.getElementById('clearHistoryButton').addEventListener('click', clearHistory);
  document.getElementById('fileInput').addEventListener('change', importHistory);

  document.getElementById('sendButton').addEventListener('click', sendRequest); 
  document.getElementById('workai').addEventListener('click', sendRequest); 


    document.getElementById('captureButton').addEventListener('click', () => {
        chrome.runtime.sendMessage({action: 'captureText'}, (response) => {
          if (response && response.text) {
            requestBody.value = response.text;
          }
        });
      });
    
      // Listen for selected text from content script
      chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
        if (message.action === 'textSelected') {
            requestBody.value = message.text;
        }
      });

}


function text_system_string(inputText,system) {
    inputText = "```" + inputText + "```"
    console.log(">>>>>>>>>>>>"+system)
    switch (system) {
        case "sendButton":
            system = "You are a helpful assistant "
            system = "```" + system + "```"
            break;
        case "workai":
            system = "work"
            system = "```" + system + "```"
            break;
        case "cardinal":
            system = "cardinal"
            system = "```" + system + "```"
            console.log("Cardinal must be in the given script");
            break;
        default:
            system = "```" + system + "```"
            console.log("Text is neither empty nor a word from the array");
            break;
    }

    // Check if inputText starts and ends with ```
    if (inputText.startsWith("```") && inputText.endsWith("```") &&  system.startsWith("```") && system.endsWith("```") ) {
        // Extract the content within the ```
        let content = inputText.slice(3, -3);
        let contentsystem = system.slice(3, -3);

        // Replace newline characters with \n and escape double quotes
        content = content.replace(/\n/g, '\\n').replace(/"/g, '\\"');
        contentsystem = contentsystem.replace(/\n/g, '\\n').replace(/"/g, '\\"');
        // Construct the final JSON object for easy comprehension
        const result = {
            text: content,
            system: contentsystem
        };

        // Return the stringified JSON
        return JSON.stringify(result);
    } else {
        // If the inputText does not start and end with ```, return a message
        return 'Invalid input: Text does not start and end with ```';
    }
}

function sendRequest(event) {

    const buttonClicked = event.target;
    const requestUrl = url.value;
/*     if (!requestUrl || !isValidUrl(requestUrl)) {
    response.value = 'Invalid URL.';
    return;
    } */
    const requestMethod = method.value; // from system
    let requestHeaders = {};
    let requestBodyContent = {};

    try {
      formatted = text_system_string(requestBody.value,buttonClicked.id);
    } catch (error) {
        console.error(error);
        response.value = 'Error parsing JSON input.';
        return;
    }
    
    if (requestMethod === 'POST' || requestMethod === 'PUT') {
        requestHeaders['Content-Type'] = 'application/json';
    }

    showLoadingSpinner();
    fetch(requestUrl, {
        method: requestMethod,
        headers: requestHeaders,
        body: (requestMethod === 'POST' || requestMethod === 'PUT') ? formatted : null, 
    })
        .then(res => {
            const contentType = res.headers.get("content-type");
            if (contentType && contentType.includes("application/json")) {
                return res.json();
            } else {
                return res.text();
            }
        })
        .then(data => {
            response.value = (typeof data === 'object') ? JSON.stringify(data, null, 2) : data;
            history.push({requestUrl, requestMethod, requestHeaders, requestBodyContent, responseData: data});
            hideLoadingSpinner();
            updateHistory();
        })
        .catch(error => {
            hideLoadingSpinner(); 
            response.value = 'Error: ' + "Local Ai unavailable";
        });
}



function isValidUrl(string) {
try {
    new URL(string);
    return true;
} catch (_) {
    return false;  
}
}

function toggleView() {
isFormattedView = !isFormattedView;
    try {
        response.value = isFormattedView ? JSON.stringify(JSON.parse(response.value), null, 2) : JSON.parse(response.value);
    } catch (error) {

        response.value = isFormattedView ? JSON.stringify(JSON.parse(response.value), null, 2) : JSON.parse(response.value);
    }
}

function exportHistory() {
const blob = new Blob([JSON.stringify(history, null, 2)], {type: 'application/json'});
const link = document.createElement('a');
link.href = URL.createObjectURL(blob);
link.download = 'history.json';
link.click();
}

function importHistory(event) {
const file = event.target.files[0];
if (file && file.type === 'application/json') {
    const reader = new FileReader();
    reader.onload = function(e) {
    try {
        history = JSON.parse(e.target.result);
        updateHistory();
    } catch (error) {
        response.value = 'Error parsing history file.';
    }
    };
    reader.readAsText(file);
} else {
    response.value = 'Invalid file type. Please select a JSON file.';
}
}

function clearHistory() {
history = [];
updateHistory();
}

function updateHistory() {
    const historyContainer = document.getElementById('historyContainer');
    historyContainer.innerHTML = history.map((entry, index) =>
      `<div class="history-entry">
         <div>${index + 1}. ${entry.requestMethod} ${entry.requestUrl}</div>
       </div>`
    ).join('');
  }

function showLoadingSpinner() {
    const loadingSpinner = document.getElementById('loadingSpinner');
    loadingSpinner.style.display = 'block';
}

function hideLoadingSpinner() {
    const loadingSpinner = document.getElementById('loadingSpinner');
    loadingSpinner.style.display = 'none';
}
  

  
