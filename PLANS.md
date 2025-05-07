# CrewAI Assistant – Project Plans

## 1. Vision & Purpose

- **Personal AI assistant** that can interact with my computer, record daily activities, and be available for ongoing conversation.
- Should be able to transcribe and summarize YouTube videos, allowing me to ask questions about their content.
- The assistant is for personal use (single user) for now.
- Help manage daily tasks, schedules, and household activities with voice alerts.
- Track and plan workouts, providing guidance and progress monitoring.

## 2. Core & Dream Features

### Core Features (Short-term)
- Voice and text interaction, always running in the background.
- Record and log daily activities via conversation.
- YouTube video transcription and summarization.
- English and Finnish language support.
- Simple memory system for:
  - Remembering plans and tasks
  - Storing important information from conversations
  - Tracking daily activities and habits
  - Quick retrieval of past information
- Daily schedule management with voice alerts
- Basic task organization and reminders
- Workout tracking and planning

### Dream Features (Long-term)
- Integration with Google Docs and other productivity tools.
- Suggest life optimizations based on activity analysis.
- Smart glasses integration for on-the-go use.
- Personalized recommendations (what to watch, do, etc.).
- Analyze my life and suggest improvements.
- Smart home integration for household management
- Automatic schedule optimization based on habits
- AI-powered workout optimization and form analysis

## 3. Integrations & Extensibility

- Planned: Google Cloud, Google Docs, YouTube API.
- Future: Add more tools as needs arise.
- Modularity: Rely on CrewAI for agent/tool modularity.

## 4. User Experience

- CLI-based for now, always-on background process.
- Future: Web interface for visualizations (e.g., graphs).
- Multilingual: English and Finnish.
- Voice alerts for important tasks and schedules.
- Voice guidance during workouts.

### Unified Mode
- [ ] (Planned) Combine voice and text modes into a single unified mode
- [ ] (Planned) Allow seamless switching between voice and text input without changing modes
- [ ] (Planned) Maintain CrewAI capabilities in both input methods
- [ ] (Planned) Automatic voice output for important notifications and responses
- [ ] (Planned) Option to toggle voice output on/off while keeping the unified mode
- [ ] (Planned) Context-aware responses that consider both voice and text history

## 5. Technical Considerations

- **Data Storage:** Start with local vector storage; consider SQL for larger datasets.
- **Performance:** Fast response for direct conversation; agent tasks can be slower.
- **Parallelism:** Explore running agents in parallel while maintaining real-time chat.
- **Privacy:** Since it's personal, focus on local storage and privacy.

## 6. Memory System

### Basic Memory Features
- [x] Simple key-value storage for quick facts and information
- [x] Conversation history with timestamps
- [x] Task and plan storage
- [ ] Daily activity logging
- [ ] Quick search and retrieval of stored information

### Memory Commands
- [ ] "Remember that..." - Store information
- [ ] "What did I say about..." - Retrieve information
- [ ] "Show me my plans for..." - View stored plans
- [ ] "What did I do yesterday..." - View activity history

### Implementation Notes
- Start with simple JSON/CSV storage
- Consider using SQLite for structured data
- Implement basic vector search for semantic retrieval
- Add timestamps to all entries
- Include metadata (context, importance, category)

### Memory System (Recent and Planned)
- [x] All conversations are logged per session in `conversation_history.txt`.
- [x] User can ask the agent (in agent mode) to review conversation history and extract important facts to memory (memories.txt) or ideas (ideas.txt).
- [x] System message instructs the assistant to respond naturally to memory requests in direct chat mode (e.g., "Sure, I'll remember that!").
- [ ] (Planned) Ability to mark "remember this" messages for easier extraction by the agent later.
- [ ] (Planned) Automatic memory extraction process that reviews conversation history at the end of the day and updates memories/ideas.

### Workout and Structured Data Logging
- [x] Free-form workout notes can be logged to `memories.txt`.
- [ ] (Planned) Separate file (e.g., `workouts.txt` or `workouts.json`) for structured workout data and analytics.

### Text-to-Speech (TTS)
- [x] Google Cloud TTS support with voice selection directly from the app.
- [x] Ability to test and preview different voices with a sample sentence.
- [ ] (Planned) Option to use other TTS services (e.g., Coqui TTS) easily.

### User Experience & File Management
- [x] `.gitignore` updated to exclude knowledge directory files and private files.
- [x] `conversation_history.txt` only logs the current session's messages.
- [x] All file paths are now absolute and based on the script location/config, so files are always created in the correct place.
- [x] All persistent data (memories, ideas, conversation history, tasks) is stored in the `knowledge/` directory at the project root.
- [x] `README.md` is up to date with correct run instructions and file locations.
- [ ] (Planned) Ability to filter and display only certain types of memories (e.g., only workouts, only ideas).

### Agent & Direct Mode
- [x] User can switch between agent mode (CrewAI) and direct mode (OpenAI API).
- [x] In agent mode, the model can use the memory tool and extract information from history.
- [x] In direct mode, memory requests are handled via system message for a natural experience.

### General & Future Ideas
- [ ] (Planned) User profile file for persistent user data (e.g., name, goals, preferences).
- [ ] (Planned) Automatic daily/periodic summary generation (e.g., "daily notes").
- [ ] (Planned) Agent can proactively ask the user if certain information should be remembered based on conversation context.

## 7. Task & Schedule Management

### Core Features
- [ ] Daily schedule creation and management
- [x] Recurring task support (currently only daily repetition is supported; other frequencies are planned)
- [ ] Priority-based task organization
- [x] Voice alerts for upcoming tasks
- [ ] Task categories (work, personal, household)
- [ ] Progress tracking and completion logging

### Household Management
- [ ] Chore tracking and rotation
- [ ] Shopping list management
- [ ] Meal planning assistance
- [ ] Maintenance schedule tracking
- [ ] Bill payment reminders

### Alert System
- [ ] Voice notifications for upcoming tasks
- [ ] Customizable alert tones
- [ ] Priority-based alert system
- [ ] Snooze functionality
- [ ] Alert history and tracking

### Commands
- [ ] "Add task: [task] at [time]"
- [ ] "Show my schedule for [day]"
- [ ] "What's next on my schedule?"
- [ ] "Add recurring task: [task] every [frequency]"
- [ ] "Add to shopping list: [item]"
- [ ] "Show me my chores for this week"

### Implementation Notes
- Use system notifications for alerts
- Store schedules in SQLite database
- Implement calendar view for schedule visualization
- Add natural language processing for task entry
- Include task templates for common activities

## 8. Workout & Fitness Management

### Core Features
- [ ] Workout plan creation and storage
- [ ] Exercise tracking with sets, reps, and weights
- [ ] Progress visualization and statistics
- [ ] Rest timer with voice alerts
- [ ] Workout history and trends
- [ ] Exercise library with descriptions
- [ ] Personal records tracking

### Workout Planning
- [ ] Create and modify workout routines
- [ ] Schedule workouts in calendar
- [ ] Track workout frequency and consistency
- [ ] Progressive overload tracking
- [ ] Rest day recommendations
- [ ] Workout split management (e.g., push/pull/legs)

### Voice Commands
- [ ] "Start workout: [workout name]"
- [ ] "Record set: [exercise] [weight] [reps]"
- [ ] "Start rest timer: [duration]"
- [ ] "Show my progress for [exercise]"
- [ ] "What's my next workout?"
- [ ] "Add exercise: [name] to [workout]"

### Implementation Notes
- Store workout data in structured format (SQLite)
- Include exercise metadata (muscle groups, equipment needed)
- Implement progress graphs and statistics
- Add voice guidance for form and timing
- Consider integration with fitness APIs for exercise data

## 9. Collaboration & Open Source

- Open source on GitHub, but not expecting outside contributors yet.

## 10. Roadmap & Milestones

### Short-term
- [ ] Reliable voice/text chat loop
- [ ] Basic memory system implementation
- [ ] Basic task and schedule management
- [ ] Voice alert system
- [ ] Basic workout tracking
- [ ] YouTube video transcription and Q&A
- [ ] Basic memory/logging of conversations

### Long-term
- [ ] Google Docs integration
- [ ] Activity analysis and life optimization suggestions
- [ ] Web dashboard for visualizations
- [ ] Smart glasses support
- [ ] Smart home integration
- [ ] Advanced schedule optimization
- [ ] Advanced workout analytics and optimization

### Deadlines
- Consider setting deadlines to encourage progress.

## 11. Risks & Challenges

- Competing projects may advance faster (but can fork if open source).
- Technical challenge: integrating many tools and keeping the system robust.
- Experimentation: Use AI for as many tasks as possible, learn what works.
- Ensuring reliable voice alerts without being intrusive.
- Maintaining accurate workout tracking during exercise.

## 12. Open Questions

- What's the best way to handle long-term memory and retrieval?
- How to balance privacy with cloud-based features?
- What are the best tools/APIs for YouTube transcription and summarization?
- How to structure memory for efficient retrieval and context awareness?
- How to implement reliable voice alerts that don't interrupt workflow?
- How to best structure workout data for analysis and progress tracking?

## Notes & Ideas

- [ ] Try out different vector databases for local storage.
- [ ] Research best practices for running always-on assistants.
- [ ] Explore open-source smart glasses projects for future integration.
- [ ] Look into simple memory systems like TinyDB or SQLite for initial implementation.
- [ ] Research existing task management systems for inspiration.
- [ ] Consider using system tray notifications for alerts.
- [ ] Look into fitness tracking apps for workout data structure inspiration.
- [ ] Research voice recognition during physical activity.
- [ ] Enable tasks.json to trigger actions, such as running CrewAI agent chains or other functions, at scheduled times (not just reminders).
- [ ] (Idea) Käyttää valmista voice chat -fronttia (esim. ChatGPT, Discord, Telegram) oman CrewAI-backendin kanssa: frontti hoitaa puheentunnistuksen ja TTS:n, oma backend CrewAI, työkalut ja muistijärjestelmä. Tämä mahdollistaa nopean käyttöönoton ilman paikallisen puheentunnistuksen ja TTS:n konfigurointia.

## Bugs & Improvements

### Keskusteluhistorian hallinta
- [ ] (Planned) Keskusteluhistorian tallennus abstrahoidaan yhteen funktioon, jota käytetään kaikissa moodeissa (voice, text, CrewAI, OpenAI), jotta historia pysyy aina ajan tasalla ja koodi pysyy DRY-periaatteen mukaisena.

### Voice-tilan korjaukset
- [ ] (Bug) Voice-tilassa keskusteluhistoria ei tallennu conversation_history.txt-tiedostoon – korjataan loggaus niin, että myös voice-tilan normaalit keskustelut tallentuvat kuten CrewAI-tilassa.

### Virheiden käsittely
- [ ] (Bug) Konsolissa esiintyy virheilmoituksia työkalujen ja tiedostojen käsittelyssä – lisätään virheenkäsittelyä ja tarkistetaan argumenttien tyypit sekä tiedostopolut.

### Suorituskyvyn optimointi
- [ ] (Planned) Lisää debug-tilan lokituksiin viiveiden (latency) mittaus: lokitetaan aika viestin lähettämisestä vastauksen alkuun sekä mahdollisuuksien mukaan eri vaiheiden (esim. API-kutsu, agentin prosessointi) kestot.
- [ ] (Planned) Viiveiden optimointi: pyritään nopeuttamaan vasteaikaa erityisesti chat-moodissa. Optimointia ja mittaustuloksia voidaan käsitellä erillisessä analyysitiedostossa (esim. PERFORMANCE.md).

---

*Update this file regularly as your ideas evolve!* 