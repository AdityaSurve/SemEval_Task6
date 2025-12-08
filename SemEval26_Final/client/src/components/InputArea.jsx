import Recorder from "./Recorder";
import { useState } from "react";

export default function InputArea({ onSubmit, disabled }) {
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState("");

  const submit = () => {
    onSubmit(question, answer);
  };

  return (
    <div className="w-full bg-slate-800/70 p-4 rounded-xl flex flex-col gap-3 border border-white/5 backdrop-blur">
      <label className="text-xs uppercase tracking-wide text-gray-400">
        Question
      </label>
      <textarea
        className="p-3 rounded-xl bg-slate-900/70 text-gray-100 border border-white/10 focus:border-teal-400/70 focus:outline-none transition"
        rows="2"
        placeholder="Enter Question..."
        value={question}
        onChange={(e) => setQuestion(e.target.value)}
      />

      <label className="text-xs uppercase tracking-wide text-gray-400">
        Answer (or use mic)
      </label>
      <div className="flex gap-3 flex-col md:flex-row">
        <textarea
          className="flex-1 p-3 rounded-xl bg-slate-900/70 text-gray-100 border border-white/10 focus:border-teal-400/70 focus:outline-none transition"
          rows="2"
          placeholder="Enter Answer or Use Mic..."
          value={answer}
          onChange={(e) => setAnswer(e.target.value)}
        />
        <Recorder onResult={(text) => setAnswer(text)} />
      </div>

      <button
        onClick={submit}
        disabled={disabled}
        className={`w-full md:w-auto px-6 py-3 rounded-xl font-bold text-white shadow-lg transition transform active:scale-95 ${
          disabled
            ? "bg-slate-700 text-gray-400 cursor-not-allowed"
            : "bg-linear-to-r from-teal-500 via-blue-500 to-purple-500 hover:shadow-teal-500/30 hover:-translate-y-px"
        }`}
      >
        {disabled ? "Analyzing..." : "Analyze Response"}
      </button>
    </div>
  );
}
