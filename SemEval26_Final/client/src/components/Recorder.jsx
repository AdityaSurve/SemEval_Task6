import { useState } from "react";
import { FaMicrophone, FaStop } from "react-icons/fa";

export default function Recorder({ onResult }) {
  const [rec, setRec] = useState(null);
  const [recording, setRecording] = useState(false);

  const start = async () => {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const recorder = new MediaRecorder(stream);
    const chunks = [];

    recorder.ondataavailable = (e) => chunks.push(e.data);
    recorder.onstop = () => {
      const blob = new Blob(chunks, { type: "audio/webm" });
      // Here you can send audio to STT service then call onResult(text)
      alert("Audio captured! (Hook up STT here)");
    };

    recorder.start();
    setRec(recorder);
    setRecording(true);
  };

  const stop = () => {
    rec.stop();
    setRecording(false);
  };

  return (
    <button
      className={`flex items-center justify-center w-12 h-12 rounded-2xl border border-white/10 shadow-md transition transform active:scale-95 ${
        recording
          ? "bg-red-500/80 hover:bg-red-500 text-white animate-pulse-soft"
          : "bg-emerald-500/80 hover:bg-emerald-500 text-white"
      }`}
      onClick={recording ? stop : start}
      title={recording ? "Stop recording" : "Record answer"}
    >
      {recording ? <FaStop /> : <FaMicrophone />}
    </button>
  );
}
