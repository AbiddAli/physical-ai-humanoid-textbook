import React, { useState, useRef, useEffect } from 'react';

const ChatModal = ({ onClose }) => {
  const [messages, setMessages] = useState([
    {
      id: 1,
      role: 'bot',
      content:
        "Hello! I'm your Physical AI assistant. Ask me anything about physical AI and humanoid robotics.",
    },
  ]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const chatContainerRef = useRef(null);
  const inputRef = useRef(null);

  // Scroll to bottom
  useEffect(() => {
    if (chatContainerRef.current) {
      chatContainerRef.current.scrollTop =
        chatContainerRef.current.scrollHeight;
    }
  }, [messages, isLoading]);

  // Focus input on mount
  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  const handleSend = async () => {
    if (!inputValue.trim() || isLoading) return;

    const userMessage = inputValue.trim();
    setInputValue('');
    setIsLoading(true);

    setMessages((prev) => [
      ...prev,
      { id: Date.now(), role: 'user', content: userMessage },
    ]);

    try {
      const response = await fetch('http://127.0.0.1:8000/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: userMessage }),
      });

      const data = await response.json();

      setMessages((prev) => [
        ...prev,
        {
          id: Date.now() + 1,
          role: 'bot',
          content: data.answer || 'No response received',
          sources: data.sources || [],
        },
      ]);
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        {
          id: Date.now() + 1,
          role: 'bot',
          content:
            'Error: Could not reach server. Please check if backend is running.',
        },
      ]);
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

  return (
    <div className="chat-modal-overlay" onClick={onClose}>
      <div
        className="chat-modal-content"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="chat-modal-header">
          <h2>Physical AI Assistant</h2>
          <button className="close-btn" onClick={onClose}>
            ×
          </button>
        </div>

        <div className="chat-messages" ref={chatContainerRef}>
          {messages.map((msg) => (
            <div
              key={msg.id}
              className={`message ${msg.role}-message`}
            >
              <strong>{msg.role === 'user' ? 'You:' : 'Bot:'}</strong>
              <div>{msg.content}</div>
            </div>
          ))}

          {isLoading && (
            <div className="message bot-message">
              <strong>Bot:</strong> typing…
            </div>
          )}
        </div>

        <div className="input-container">
          <input
            ref={inputRef}
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="Ask your question…"
            disabled={isLoading}
          />
          <button
            onClick={handleSend}
            disabled={isLoading || !inputValue.trim()}
          >
            Send
          </button>
        </div>
      </div>
    </div>
  );
};

export default ChatModal;
