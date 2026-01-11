import React from 'react';
import Layout from '@theme/Layout';
import Link from '@docusaurus/Link';
import Head from '@docusaurus/Head';

export default function Home() {
  return (
    <Layout
      title="Physical AI Humanoid Textbook"
      description="A structured, industry-grade curriculum for Physical AI and Humanoid Robotics"
    >
      <Head>
        <link rel="stylesheet" href="/chat-modal.css" />
        <script src="/chat-modal.js" defer></script>
      </Head>

      {/* HERO SECTION */}
      <header
        style={{
          padding: '6rem 2rem',
          textAlign: 'center',
          background: 'linear-gradient(135deg, #1e3c72, #2a5298)',
          color: '#ffffff',
        }}
      >
        <h1 style={{ fontSize: '3rem', marginBottom: '1rem' }}>
          Physical AI & Humanoid Robotics
        </h1>
        <p
          style={{
            fontSize: '1.3rem',
            maxWidth: '900px',
            margin: '0 auto',
            opacity: 0.95,
          }}
        >
          A structured, week-by-week textbook covering ROS 2, digital twins,
          industrial simulation, and multimodal intelligence for humanoid robots.
        </p>

        <div style={{ marginTop: '2.5rem' }}>
          <Link
            className="button button--primary button--lg"
            to="/docs/week-01/module-01-01/chapter-01-introduction"
            style={{ marginRight: '1rem' }}
          >
            📘 Start Learning
          </Link>

          <button
            className="button button--outline button--secondary button--lg"
            onClick={() => window.openChatModal && window.openChatModal()}
            style={{ cursor: 'pointer' }}
          >
            🤖 AI Tutor
          </button>
        </div>
      </header>

      {/* MAIN CONTENT */}
      <main style={{ padding: '4rem 2rem', maxWidth: '1100px', margin: 'auto' }}>
        {/* WHAT THIS BOOK IS */}
        <section style={{ marginBottom: '4rem' }}>
          <h2>📚 What This Textbook Covers</h2>
          <ul style={{ fontSize: '1.1rem', lineHeight: '1.8' }}>
            <li>Foundations of Physical AI and embodied intelligence</li>
            <li>ROS 2 as the robotic nervous system</li>
            <li>Digital twins using Gazebo and Unity</li>
            <li>NVIDIA Isaac Sim and Isaac ROS</li>
            <li>Vision-Language-Action (VLA) systems</li>
            <li>End-to-end humanoid robot capstone project</li>
          </ul>
        </section>

        {/* STRUCTURE */}
        <section style={{ marginBottom: '4rem' }}>
          <h2>🗂 Curriculum Structure</h2>
          <p style={{ fontSize: '1.1rem' }}>
            The curriculum is organized into <strong>weeks and modules</strong>,
            following a progressive learning path used in real-world robotics
            teams and research labs.
          </p>
          <p style={{ fontSize: '1.1rem' }}>
            Each week introduces core concepts, simulation workflows, and
            applied intelligence patterns for humanoid robots.
          </p>
        </section>

        {/* ROADMAP */}
        <section>
          <h2>🛣 Roadmap</h2>

          <h3>✅ Implemented</h3>
          <ul style={{ fontSize: '1.05rem', lineHeight: '1.7' }}>
            <li>Weeks 1–6 core textbook curriculum</li>
            <li>Professional documentation site</li>
            <li>Simulation-first humanoid learning path</li>
          </ul>

          <h3 style={{ marginTop: '1.5rem' }}>🚀 Planned (Hackathon Roadmap)</h3>
          <ul style={{ fontSize: '1.05rem', lineHeight: '1.7' }}>
            <li>Weeks 7–10: Manipulation, control, and reinforcement learning</li>
            <li>Weeks 11–13: Deployment, safety, ethics, and evaluation</li>
            <li>Multilingual textbook translation</li>
            <li>Embedded AI Tutor powered by Qwen + RAG</li>
          </ul>
        </section>
      </main>

      {/* Chat Modal */}
      <div id="chat-modal" className="chat-modal" style={{ display: 'none' }}>
        <div className="chat-modal-content">
          <div className="chat-modal-header">
            <h2>Physical AI Assistant</h2>
            <button className="close-btn">&times;</button>
          </div>
          <div className="chat-container">
            <div id="chat" className="chat-messages"></div>
            <div className="input-container">
              <input
                type="text"
                id="query"
                placeholder="Ask your question..."
                autoComplete="off"
              />
              <button id="send-btn">Send</button>
            </div>
          </div>
        </div>
      </div>
    </Layout>
  );
}
