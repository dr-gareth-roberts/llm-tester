# LLM Tester
A comprehensive Python library for the systematic investigation of large language models' performance profile. 

## About

A large collection of functions allow for user's to assemble their own scripts to test model consistency, dynamically generate best performing prompts via optimisation of prompt structuring/ordering/complexity, determine whether prompt modifications could lead to better cost optimisation, perform exhaustive parameter sweeps across different LLMs, and many more optimisation scenarios.

## What can it do?

The library is a powerful tool for AI engineers and researchers interested in quantifying things such as:
- topic consistency checks (via similarity of adjacent sentence embedding values)
- lexical and synonymic cohesion (via synonym usage and lexical variations within a topic)
- sequential coherence analysis (does the LLM output show logical progression across different sections/paragraphs of the output)
- readability metrics (Flesch reading ease, Gunning Fog Index, SMOG index, etc) and detailed grammar analysis (subject-verb agreement issues, tense inconsistencies, punctuation errors, etc)
- exploration-exploitation balance (can the LLM will explore new ideas without contradicting itself via inference of LLM output diversity and self-contradiction detection)
- relevance-stability ratio (does the model stay on topic without excessive repetition)
- bias amplification/reduction testing (do certain prompts amplify biases, are biases introduced in early prompts maintained or even amplified over conversation turns?)
- context manipulations (reintegration speed, prompt perturbation sensitivity, resilience to frequent context switching)
- evaluation of reasoning chains (including determining recursive reasoning depth, the logic fidelity during the chain, correct detection of logical fallacies, etc.).

Finally, the library includes a lot of theoretical models on how value alignment and corrigibility processes may be implemented in commercial LLMs.


<img width="794" alt="image" src="https://github.com/user-attachments/assets/4c1dd6a5-3f86-4b49-ad52-d851b3096e1f">



Here are simulations of a user requesting CAPTCHA breaking knowledge. Initially, the model determines it to be high risk, until recognition of user intent raises through context provided by the user. Namely that he runs a high-traffic website and is suffering from regular denial-of-service attacks. His knowledge of fundamental security concepts and professional expertise in describing how the attacks are bypassing the CAPTCHA system he has implemented leads the model to deem his initial high-risk query as safe, and sets the scope of its response. The interaction between user and LLM finishes  with all processes in the safe threshold zone.

<img width="722" alt="image" src="https://github.com/user-attachments/assets/42f1c2db-0ed2-49c0-b6a0-47056f5f8f0e">
