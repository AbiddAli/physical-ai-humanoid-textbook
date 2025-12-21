# Project Constitution
## Physical AI & Humanoid Robotics — AI-Native Textbook

### 1. Purpose
This project defines the authoritative standards for building an AI-native
textbook and learning platform for the Physical AI & Humanoid Robotics course.

The goal is to teach embodied intelligence: AI systems that perceive, reason,
and act in the physical world.

---

### 2. Core Principles (Non-Negotiable)

#### CP-1 Technical Accuracy
All content must be technically correct and verified against official sources
(ROS 2, NVIDIA Isaac, Gazebo, Unity, OpenAI).

Hallucinated APIs or unverifiable claims are forbidden.

#### CP-2 Educational Clarity
The writing must be suitable for a Computer Science capstone audience.
Each concept must follow:
Concept → Example → Application.

#### CP-3 Embodied AI First
Digital AI concepts are allowed only if they directly support physical embodiment,
robot perception, control, or interaction.

#### CP-4 Architectural Minimalism
Use the minimum viable technology stack.
Avoid unnecessary frameworks and abstractions.

#### CP-5 Free-Tier Viability
All components must run within free-tier limits of external services.

---

### 3. Platform Standards

- Markdown / MDX only
- Docusaurus documentation framework
- Public deployment via GitHub Pages or Vercel

---

### 4. RAG Standards

- Closed-domain only (book content)
- Must support answering from user-selected text
- Mandatory stack:
  - FastAPI
  - OpenAI Agents / ChatKit SDKs
  - Neon Serverless Postgres
  - Qdrant Cloud (Free Tier)

---

### 5. Content Coverage

The book must fully cover:
- ROS 2 (Robotic Nervous System)
- Gazebo & Unity (Digital Twin)
- NVIDIA Isaac (AI Robot Brain)
- Vision-Language-Action (VLA)
- Capstone Autonomous Humanoid

---

### 6. Ethics & Originality

- Zero plagiarism tolerance
- All text must be original
- AI assistance must be reviewed and validated

---

### 7. Success Criteria

The project is successful only if:
- Book is publicly deployed
- RAG chatbot is functional
- Selected-text QA works
- All modules are covered
