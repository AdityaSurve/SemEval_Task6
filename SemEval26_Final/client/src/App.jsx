import { useState } from "react";
import ChatBubble from "./components/ChatBubble";
import InputArea from "./components/InputArea";

export default function App() {
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);

  const sendToAPI = async (question, answer) => {
    if (!question.trim() || !answer.trim() || loading) return;

    const loadingId = Date.now();
    setLoading(true);
    setMessages((prev) => [
      ...prev,
      { sender: "user", text: `Q: ${question}` },
      { sender: "user", text: `A: ${answer}` },
      { sender: "bot", loading: true, id: loadingId },
    ]);

    try {
      const res = await fetch("http://localhost:5000/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, answer }),
      });

      const data = await res.json();

      setMessages((prev) => [
        ...prev.filter((m) => !m.loading),
        { sender: "bot", text: `🧠 Clarity: ${data.predictions.clarity}` },
        { sender: "bot", text: `🎯 Evasion: ${data.predictions.evasion}` },
      ]);
    } catch (error) {
      setMessages((prev) => [
        ...prev.filter((m) => !m.loading),
        {
          sender: "bot",
          text: "⚠️ Something went wrong. Please try again.",
          error: true,
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-linear-to-b from-slate-950 via-slate-900 to-slate-950 text-gray-100">
      <div className="max-w-5xl mx-auto py-10 px-4 md:px-8">
        <header className="text-center mb-8">
          <p className="text-sm uppercase tracking-[0.3em] text-teal-300/80">
            SemEval 2026
          </p>
          <h1 className="text-4xl md:text-5xl font-black mt-2 bg-linear-to-r from-teal-300 via-blue-400 to-purple-500 bg-clip-text text-transparent drop-shadow-lg">
            Political Answer Analyzer
          </h1>
          <p className="text-gray-300 font-semibold mt-3 max-w-2xl mx-auto">
            Paste a question and an answer to gauge clarity and evasion. Smooth
            UI, fast feedback, and ambient animations to keep you in flow.
          </p>
        </header>

        <div className="glass-card w-full rounded-2xl shadow-2xl border border-white/10 backdrop-blur-xl overflow-hidden">
          <div className="h-[40vh] overflow-y-auto px-5 pb-6 pt-5 space-y-2 bg-linear-to-b from-slate-900/70 to-slate-950/70">
            {messages.length === 0 ? (
              <div className="text-center text-gray-400 py-10 animate-fade-in">
                No messages yet. Drop a question and answer to begin.
              </div>
            ) : (
              messages.map((m, i) => (
                <ChatBubble
                  key={m.id ?? i}
                  sender={m.sender}
                  text={m.text}
                  loading={m.loading}
                  error={m.error}
                />
              ))
            )}
          </div>

          <div className="border-t border-white/10 bg-slate-900/70 p-4">
            <InputArea onSubmit={sendToAPI} disabled={loading} />
          </div>
        </div>
      </div>
    </div>
  );
}
