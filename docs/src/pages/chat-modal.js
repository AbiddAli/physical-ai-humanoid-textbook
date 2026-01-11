// Chat functionality for AI Tutor modal
let chatHistory = [];  // Store chat history

function initializeChatModal() {
  const modal = document.getElementById('chat-modal');
  const closeBtn = document.querySelector('.close-btn');
  const sendBtn = document.getElementById('send-btn');
  const queryInput = document.getElementById('query');
  const chatDiv = document.getElementById('chat');

  // Check if all required elements exist
  if (!modal || !closeBtn || !sendBtn || !queryInput || !chatDiv) {
    console.error('One or more required chat elements not found');
    return;
  }

  // Close modal when close button is clicked
  closeBtn.addEventListener('click', () => {
    modal.style.display = 'none';
  });

  // Close modal when clicking outside the content
  window.addEventListener('click', (e) => {
    if (e.target === modal) {
      modal.style.display = 'none';
    }
  });

  // Handle send button click
  sendBtn.addEventListener('click', sendQuery);

  // Handle Enter key press
  queryInput.addEventListener('keypress', function(e) {
    if (e.key === 'Enter') {
      sendQuery();
    }
  });

  // Function to send query to backend
  async function sendQuery() {
    const query = queryInput.value.trim();
    if (!query) return;

    // Disable input while processing
    queryInput.disabled = true;
    sendBtn.disabled = true;

    // Add user message to chat window and history
    addMessageToChat('user', query);
    chatHistory.push({ role: 'user', content: query });
    queryInput.value = '';

    // Add typing indicator
    const typingIndicator = document.createElement('div');
    typingIndicator.className = 'typing-indicator';
    typingIndicator.innerText = 'Bot is typing...';
    chatDiv.appendChild(typingIndicator);
    chatDiv.scrollTop = chatDiv.scrollHeight;

    try {
      const res = await fetch('http://127.0.0.1:8000/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          question: query
        })
      });

      if (!res.ok) {
        throw new Error(`HTTP error! status: ${res.status}`);
      }

      const data = await res.json();

      // Remove typing indicator
      typingIndicator.remove();

      // Add bot response to chat window and history
      addMessageToChat('bot', data.answer || 'No response received', data.sources || []);
      chatHistory.push({ role: 'bot', content: data.answer || 'No response received', sources: data.sources || [] });

      // Scroll to bottom
      chatDiv.scrollTop = chatDiv.scrollHeight;
    } catch (error) {
      console.error('Error:', error);

      // Remove typing indicator
      typingIndicator.remove();

      // Show error message
      const errorMessage = document.createElement('div');
      errorMessage.className = 'message bot-message';
      errorMessage.innerHTML = `<strong>Error:</strong> Could not reach server. Please check if the backend is running.<br><small>Error details: ${error.message}</small>`;
      chatDiv.appendChild(errorMessage);
      chatDiv.scrollTop = chatDiv.scrollHeight;
    } finally {
      // Re-enable input
      queryInput.disabled = false;
      sendBtn.disabled = false;
      queryInput.focus();
    }
  }

  // Function to add message to chat
  function addMessageToChat(role, content, sources = []) {
    const chatDiv = document.getElementById('chat');
    if (!chatDiv) {
      console.error('Chat container element not found');
      return;
    }

    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role}-message`;

    if (role === 'user') {
      messageDiv.innerHTML = `<strong>You:</strong> ${content}`;
    } else {
      messageDiv.innerHTML = `<strong>Bot:</strong> ${content}`;

      // Add sources if available
      if (sources && Array.isArray(sources) && sources.length > 0) {
        const sourcesDiv = document.createElement('div');
        sourcesDiv.className = 'message-sources';

        const sourcesTitle = document.createElement('div');
        sourcesTitle.className = 'sources-title';
        sourcesTitle.textContent = 'Sources:';

        sourcesDiv.appendChild(sourcesTitle);

        sources.forEach((source, index) => {
          const sourceItem = document.createElement('div');
          sourceItem.className = 'source-item';

          // Handle different source formats that might come from the backend
          if (typeof source === 'string') {
            // If source is just a string
            sourceItem.textContent = source;
          } else if (source.url) {
            // If source has a URL
            const link = document.createElement('a');
            link.href = source.url;
            link.target = '_blank';
            link.className = 'source-link';
            link.textContent = source.title || source.url;
            sourceItem.appendChild(link);
          } else if (source.title) {
            // If source has a title
            sourceItem.textContent = source.title;
          } else if (source.content) {
            // If source has content
            sourceItem.textContent = source.content.substring(0, 100) + (source.content.length > 100 ? '...' : '');
          } else {
            // Fallback
            sourceItem.textContent = `Source ${index + 1}`;
          }

          sourcesDiv.appendChild(sourceItem);
        });

        messageDiv.appendChild(sourcesDiv);
      }
    }

    chatDiv.appendChild(messageDiv);
    chatDiv.scrollTop = chatDiv.scrollHeight; // Ensure scroll to bottom after adding message
  }

  // Initialize with a welcome message
  if (chatDiv && chatDiv.children.length === 0) {
    const welcomeMessage = document.createElement('div');
    welcomeMessage.className = 'message bot-message';
    welcomeMessage.innerHTML = 'Hello! I\'m your Physical AI assistant. Ask me anything about physical AI and humanoid robotics.';
    chatDiv.appendChild(welcomeMessage);
  }
}

// Function to open the chat modal
function openChatModal() {
  const modal = document.getElementById('chat-modal');
  if (modal) {
    modal.style.display = 'flex';
  }
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', initializeChatModal);