export default function ChatBubble({ sender, text, loading, error }) {
  const isBot = sender === "bot";
  const baseBubble =
    "max-w-[80%] px-4 py-3 rounded-2xl shadow-xl animate-fade-in-up";

  const colorBubble = loading
    ? "bg-slate-800/80 border border-teal-400/30 text-teal-200"
    : error
      ? "bg-red-500/10 border border-red-400/40 text-red-200"
      : isBot
        ? "bg-slate-800/80 border border-blue-400/30 text-blue-100"
        : "bg-gradient-to-r from-blue-500 via-indigo-500 to-purple-500 text-white";

  return (
    <div className={`w-full flex ${isBot ? "justify-start" : "justify-end"} mb-3`}>
      <div className={`${baseBubble} ${colorBubble}`}>
        {loading ? (
          <div className="flex items-center gap-2">
            <div className="loader-dot" />
            <div className="loader-dot delay-150" />
            <div className="loader-dot delay-300" />
            <span className="text-sm opacity-80">Analyzing...</span>
          </div>
        ) : (
          text
        )}
      </div>
    </div>
  );
}
