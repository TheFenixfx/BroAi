
function getUserBookmarks() {
    chrome.bookmarks.getTree(function (bookmarkTreeNodes) {
        displayUserBookmarks(bookmarkTreeNodes);
    });
}

function displayUserBookmarks(bookmarkTreeNodes) {
    const bookmarksSection = document.getElementById('bookmarksSection');
    let bookmarksHtml = '';
    bookmarkTreeNodes.forEach(processNode);

    function processNode(node) {
        if (node.children) {
            node.children.forEach(processNode);
        } else {
            bookmarksHtml += `<div><a href="${node.url}" target="_blank">${node.title}</a></div>`;
        }
    }

    bookmarksSection.innerHTML = bookmarksHtml;
}



