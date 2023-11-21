// Log when the extension's background script is loaded
console.log('Background script loaded.');

// Listen for the installation event to log a message

 chrome.runtime.onInstalled.addListener(function() {
  console.log('API Requester Extension installed.');
}); 


let selectedText = '';

 chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  async function handleTextSelected() {
    console.log(message)
    if (message.action === 'textSelected') {
       selectedText = message.text;       //const selectedText = message.text;
     }
     if (message.action === 'captureText') {
      sendResponse({text: selectedText});
      console.log('Received Selected Text:', selectedText);
      try {
        // Example: const someData = await someAsyncFunction(selectedText);
        cachex = selectedText;
        sendResponse({status: 'success',text:selectedText}); // Send a success response
      } catch (error) {
        console.error('An error occurred:', error);
        sendResponse({status: 'error', error: error.toString()}); // Send an error response
      }
    }
  }
  
  handleTextSelected()
  .then(
    message => {
        console.log("checking then"+message.text)
  }
    
    /* () => {} */) // Necessary to indicate that the response will be sent asynchronously
  .catch(() => {}); // To handle any uncaught errors

  return true; // Indicates that the response will be sent asynchronously
});
 

