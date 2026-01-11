// Chat functionality for AI Tutor modal
window.chatHistory = window.chatHistory || [];  // Store chat history

function initializeChatModal() {
  // Prevent multiple initializations
  if (window.chatModalInitialized) {
    return;
  }

  window.chatModalInitialized = true;

  const modal = document.getElementById('chat-modal');
  const closeBtn = document.querySelector('.close-btn');
  const sendBtn = document.getElementById('send-btn');
  const queryInput = document.getElementById('query');
  const chatDiv = document.getElementById('chat');

  // Close modal when close button is clicked
  if (!closeBtn.hasAttribute('data-listener-added')) {
    closeBtn.addEventListener('click', () => {
      modal.style.display = 'none';
    });
    closeBtn.setAttribute('data-listener-added', 'true');
  }

  // Close modal when clicking outside the content
  if (!window.chatClickListenerAdded) {
    window.addEventListener('click', (e) => {
      if (e.target === modal) {
        modal.style.display = 'none';
      }
    });
    window.chatClickListenerAdded = true;
  }

  // Handle send button click
  if (!sendBtn.hasAttribute('data-listener-added')) {
    sendBtn.addEventListener('click', sendQuery);
    sendBtn.setAttribute('data-listener-added', 'true');
  }

  // Handle Enter key press
  if (!queryInput.hasAttribute('data-listener-added')) {
    queryInput.addEventListener('keypress', function(e) {
      if (e.key === 'Enter') {
        sendQuery();
      }
    });
    queryInput.setAttribute('data-listener-added', 'true');
  }

  // Function to send query to backend
  async function sendQuery() {
    const query = queryInput.value.trim();
    if (!query) return;

    // Disable input while processing
    queryInput.disabled = true;
    sendBtn.disabled = true;

    // Add user message to chat window and history
    addMessageToChat('user', query);
    window.chatHistory.push({ role: 'user', content: query });
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

      // Process the response to handle weak answers
      let processedAnswer = data.answer || 'No response received';
      let processedSources = data.sources || [];

      // Check if the response indicates insufficient context
      if (processedAnswer.toLowerCase().includes('not explicitly provided') ||
          processedAnswer.toLowerCase().includes('no relevant context') ||
          processedAnswer.toLowerCase().includes('answer is not provided')) {
        // Enhance weak response with a suggestion
        processedAnswer += " I couldn't find detailed information about this topic in the current materials. Would you like me to explain the concept more generally or help you navigate to relevant sections?";
      }

      // Add bot response to chat window and history
      addMessageToChat('bot', processedAnswer, processedSources);
      window.chatHistory.push({ role: 'bot', content: processedAnswer, sources: processedSources });

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
  if (chatDiv.children.length === 0) {
    const welcomeMessage = document.createElement('div');
    welcomeMessage.className = 'message bot-message';
    welcomeMessage.innerHTML = 'Hello! I\'m your Physical AI assistant. Ask me anything about physical AI and humanoid robotics.';
    chatDiv.appendChild(welcomeMessage);
    chatDiv.scrollTop = chatDiv.scrollHeight; // Ensure scroll to bottom after adding welcome message
  }
}

// Function to open the chat modal
function openChatModal() {
  const modal = document.getElementById('chat-modal');
  if (modal) {
    modal.style.display = 'flex';
  }
}

// Initialize when DOM is loaded - ensure it only runs once
if (!window.chatDOMInitialized) {
  document.addEventListener('DOMContentLoaded', initializeChatModal);
  window.chatDOMInitialized = true;
}

// Also try to initialize immediately in case DOM is already loaded
if (document.readyState === 'loading') {
  // Still loading, DOMContentLoaded will handle it
} else {
  // Already loaded, initialize now
  setTimeout(initializeChatModal, 0); // Use timeout to ensure other scripts run first
}