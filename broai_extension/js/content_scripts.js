
document.addEventListener('mouseup', () => {
  const selectedText = window.getSelection().toString().trim();
  if (selectedText.length > 0) {
    //console.log('Selected Text:', selectedText);
    (async () => {
      try {
        const response = await chrome.runtime.sendMessage({action: "textSelected", text: selectedText});
        if (response.status === 'success') {
          console.log('granted');
          //console.log(response.text);
          //console.log('------>'+ response.text);
        } else {
          console.log('not granted');
        }
      } catch (error) {
        console.error('An error occurred:', error);
      }
    })();
  }
});
