import React, { useState, useRef, useEffect } from 'react';

const ChatModal = ({ isOpen, onClose }) => {
  const [messages, setMessages] = useState([
    { id: 1, role: 'bot', content: 'Hello! I\'m your Physical AI assistant. Ask me anything about physical AI and humanoid robotics.' }
  ]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const chatContainerRef = useRef(null);
  const inputRef = useRef(null);

  // Scroll to bottom of chat when messages change
  useEffect(() => {
    if (chatContainerRef.current) {
      chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight;
    }
  }, [messages]);

  // Focus input when modal opens
  useEffect(() => {
    if (isOpen && inputRef.current) {
      inputRef.current.focus();
    }
  }, [isOpen]);

  const handleSend = async () => {
    if (!inputValue.trim() || isLoading) return;

    const userMessage = inputValue.trim();
    setInputValue('');

    // Add user message to chat
    const userMessageObj = {
      id: Date.now(),
      role: 'user',
      content: userMessage
    };

    setMessages(prev => [...prev, userMessageObj]);
    setIsLoading(true);

    try {
      // Send request to backend
      const response = await fetch('http://127.0.0.1:8000/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ question: userMessage })
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();

      // Add bot response to chat
      const botMessageObj = {
        id: Date.now() + 1,
        role: 'bot',
        content: data.answer || 'No response received',
        sources: data.sources || []
      };

      setMessages(prev => [...prev, botMessageObj]);
    } catch (error) {
      console.error('Error sending message:', error);

      // Add error message to chat
      const errorMessageObj = {
        id: Date.now() + 1,
        role: 'bot',
        content: `Error: Could not reach server. Please check if the backend is running at http://127.0.0.1:8000/chat`
      };

      setMessages(prev => [...prev, errorMessageObj]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const renderMessageContent = (content, sources) => {
    return (
      <div>
        <div><strong>{content}</strong></div>
        {sources && sources.length > 0 && (
          <div className="message-sources">
            <div className="sources-title">Sources:</div>
            {sources.map((source, index) => (
              <div key={index} className="source-item">
                {typeof source === 'string' ? (
                  <div>{source}</div>
                ) : source.url ? (
                  <a href={source.url} target="_blank" rel="noopener noreferrer" className="source-link">
                    {source.title || source.url}
                  </a>
                ) : source.title ? (
                  <div>{source.title}</div>
                ) : source.content ? (
                  <div>{source.content.substring(0, 100)}{source.content.length > 100 ? '...' : ''}</div>
                ) : (
                  <div>Source {index + 1}</div>
                )}
              </div>
            ))}
          </div>
        )}
      </div>
    );
  };

  if (!isOpen) return null;

  return (
    <div className="chat-modal" onClick={onClose}>
      <div className="chat-modal-content" onClick={e => e.stopPropagation()}>
        <div className="chat-modal-header">
          <h2>Physical AI Assistant</h2>
          <button className="close-btn" onClick={onClose}>&times;</button>
        </div>
        <div className="chat-container">
          <div className="chat-messages" ref={chatContainerRef}>
            {messages.map((message) => (
              <div
                key={message.id}
                className={`message ${message.role}-message`}
              >
                <div><strong>{message.role === 'user' ? 'You:' : 'Bot:'}</strong> </div>
                {renderMessageContent(message.content, message.sources)}
              </div>
            ))}
            {isLoading && (
              <div className="message bot-message">
                <div><strong>Bot:</strong> Bot is typing...</div>
              </div>
            )}
          </div>
          <div className="input-container">
            <input
              ref={inputRef}
              type="text"
              id="query"
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="Ask your question..."
              autoComplete="off"
              disabled={isLoading}
            />
            <button
              id="send-btn"
              onClick={handleSend}
              disabled={isLoading || !inputValue.trim()}
            >
              Send
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ChatModal;