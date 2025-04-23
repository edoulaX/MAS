
LangGraph vs LangChain: 
- https://www.youtube.com/watch?v=qAF1NjEVHhY&t=19s

\u{2764}:
- https://arxiv.org/pdf/2402.03578 

- LLm agent: A survey on Methodology, Applications and challenges
https://arxiv.org/abs/2503.21460

- LLMs Working in Harmony: A Survey on the Technological Aspects of Building Effective LLM-Based Multi Agent Systems
https://arxiv.org/pdf/2504.01963


Intro, Overview, Planning, Agent Memory, Applications



Intro, Agent Methodology, Evaluation and tools, Applications, Implementation, Results, Discussion (Benefits/Limitation Real-World Issues,, ethics), Conclusion


= 1. Introduction
- Overview of LLM agents and their significance in AI.
- Comparison with traditional AI systems.
- Key developments driving LLM agents: reasoning capabilities, tool manipulation, and memory architectures.
- Objectives and contributions.

= 2. Agent Methodology
- *2.1 Agent Construction*
  - *2.1.1 Profile Definition*
    - Human-Curated Static Profiles
    - Batch-Generated Dynamic Profiles
  - *2.1.2 Memory Mechanism*
    - Short-Term Memory
    - Long-Term Memory
    - Knowledge Retrieval as Memory
    - Enhancing agents with long-term and contextual memory
  - *2.1.3 Planning Capability*
    - Task Decomposition Strategies
    - Feedback-Driven Iteration
    - Strategies for long-term task decomposition and goal achievement
    - Using algorithms like tree search to guide reasoning and decision-making
  - *2.1.4 Action Execution*
    - Tool Utilization:
      - used by LLM agents
      - created by LLM agents
    - Physical Interaction
- *2.2 Agent Collaboration/ Architecture*
  - *2.2.1 Centralized Control*
  - *2.2.2 Decentralized Collaboration*
  - *2.2.3 Hybrid Architecture*
  - *2.2.4 A2A*
- *2.3 Infrastructure for deploying LLM agents*
  - Autogen, CrewAi, LangChain/graph/smith ...
  - Model context Protocol (MCP)
  - A2A (agent2agent)

= 3. Evaluation and Techniques for Enhancement
- *3.1 Evaluation Benchmarks and Datasets*
  - *3.1.1 General Assessment Frameworks*
  - *3.1.2 Domain-Specific Evaluation System*
  - *3.1.3 Collaborative Evaluation of Complex Systems*
- *3.2 Techniques for Enhancement*
  - Feedback & Reflection: Incorporating self-evaluation and external feedback
  - RAG (Retrieval-Augmented Generation): Integrating external knowledge sources
  - RCAG
  - Autonomous Optimization and Self-Learning
    - Self-Supervised Learning
    - Self-Reflection and Self-Correction
    - Self-Rewarding and Reinforcement Learning
  - Multi-Agent Co-Evolution
    - Cooperative and Collaborative Learning
    - Competitive and Adversarial Co-Evolution
  - Evolution via External Resources
    - Knowledge-Enhanced Evolution
    - External Feedback-Driven Evolution

= 4. Training Strategies for LLM agents
  - *4.1 Fine-tuning*: Domain-specific model customization.
  - *4.2 RL (Reinforcement Learning)*: Training via interaction and reward signals.
  - *4.3 DPO (Direct Preference Optimization)*: Training from human or machine preferences.

= 5. Applications
- *5.1 Scientific Discovery*
  - Agentic AI Across Scientific Disciplines
  - Agentic AI in Chemistry, Materials Science, and Astronomy
  - Agentic AI in Biology
  - Agentic AI in Scientific Dataset Construction
  - Agentic AI in Medical
- *5.2 Gaming*
  - Game Playing
  - Game Generation
- *5.3 Social Science*
  - Economy
  - Psychology
  - Social Simulation
- *5.4 Productivity Tools*
  - Software Development
  - Recommender Systems

= 6. Implementation
- *6.1 Manufacturing use case*

= 7. Result

= 8. Real-World Issues
- *8.1 Agent-centric Security*
  - *8.1.1 Adversarial Attacks and Defense*
  - *8.1.2 Jailbreaking Attacks and Defense*
  - *8.1.3 Backdoor Attacks and Defense*
  - *8.1.4 Model Collaboration Attacks and Defense*
- *8.2 Data-centric Security*
  - *8.2.1 External Data Attack and Defense*
  - *8.2.2 Interaction Attack and Defense*
- *8.3 Privacy*
  - *8.3.1 LLM Memorization Vulnerabilities*
  - *8.3.2 LLM Intellectual Property Exploitation*
- *8.4 Social Impact and Ethical Concerns*
  - *8.4.1 Benefits to Society*
  - *8.4.2 Ethical Concerns*

= 9. Challenges and Future Trends
- Scalability and Coordination
- Stability
- Memory Constraints and Long-Term Adaptation
- Stability/Reliability and Scientific Rigor
    - *Safety*: Avoiding harmful outputs.
    - *Bias*: Mitigating social and data biases.
    - *Hallucination*: Reducing false information generation.
- Multi-turn, Multi-agent Dynamic Evaluation
- Role-playing Scenarios
- Emerging Technologies that could impact LLM agent development (quantum computing or neuromorphic engineering)

= 10. Conclusion
- Summary of contributions and findings.
- Future directions and potential advancements in LLM agent technologies.
- Potential personal use case

= References
- Comprehensive list of references cited in the paper.

