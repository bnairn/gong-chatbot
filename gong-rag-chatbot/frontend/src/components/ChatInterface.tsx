import { useState, useRef, useEffect } from 'react';
import { Send, Bot, User, Loader2, ChevronDown, ChevronUp, Sparkles, MessageSquare } from 'lucide-react';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const exampleQueries = [
  "What are the most common use cases mentioned by prospective customers?",
  "Which competitors are most frequently mentioned in calls?",
  "What are the top objections to using Port?",
  "Which user personas are most involved from the customer side?",
  "What common themes do customers bring up?",
  "What are the top 5 industries in recent calls?"
];

interface Message {
  role: 'user' | 'assistant';
  content: string;
  sources?: Source[];
  queryTime?: number;
  error?: boolean;
}

interface Source {
  call_id: string;
  call_title: string;
  call_date: string;
  customer_company?: string;
  relevance_score: number;
  excerpt: string;
}

export default function ChatInterface() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [expandedSources, setExpandedSources] = useState<Record<number, boolean>>({});
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => { messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' }); }, [messages]);

  const sendMessage = async (text: string) => {
    if (!text.trim() || loading) return;
    const userMsg: Message = { role: 'user', content: text };
    setMessages(prev => [...prev, userMsg]);
    setInput('');
    setLoading(true);
    try {
      const history = messages.map(m => ({ role: m.role, content: m.content }));
      const res = await fetch(`${API_URL}/api/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: text, conversation_history: history })
      });
      if (!res.ok) throw new Error('Failed');
      const data = await res.json();
      setMessages(prev => [...prev, { role: 'assistant', content: data.answer, sources: data.sources, queryTime: data.query_time_ms }]);
    } catch {
      setMessages(prev => [...prev, { role: 'assistant', content: 'Sorry, I encountered an error. Please try again.', error: true }]);
    } finally { setLoading(false); }
  };

  return (
    <div className="flex flex-col h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900">
      <header className="bg-slate-800/50 backdrop-blur-sm border-b border-slate-700/50 px-6 py-4">
        <div className="max-w-4xl mx-auto flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-violet-500 to-indigo-600 flex items-center justify-center shadow-lg shadow-violet-500/25">
            <Bot className="w-6 h-6 text-white" />
          </div>
          <div>
            <h1 className="text-xl font-semibold text-white">Call Insights AI</h1>
            <p className="text-sm text-slate-400">Powered by Gong transcript analysis</p>
          </div>
        </div>
      </header>
      <main className="flex-1 overflow-y-auto px-4 py-6">
        <div className="max-w-4xl mx-auto space-y-6">
          {messages.length === 0 && (
            <div className="text-center py-12">
              <div className="w-16 h-16 mx-auto mb-6 rounded-2xl bg-gradient-to-br from-violet-500/20 to-indigo-600/20 flex items-center justify-center border border-violet-500/30">
                <Sparkles className="w-8 h-8 text-violet-400" />
              </div>
              <h2 className="text-2xl font-semibold text-white mb-2">Ask about your sales calls</h2>
              <p className="text-slate-400 mb-8 max-w-md mx-auto">Get insights from customer conversations.</p>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3 max-w-2xl mx-auto">
                {exampleQueries.map((q, i) => (
                  <button key={i} onClick={() => sendMessage(q)} className="text-left px-4 py-3 rounded-xl bg-slate-800/50 border border-slate-700/50 hover:border-violet-500/50 hover:bg-slate-700/50 transition-all text-sm text-slate-300 hover:text-white">
                    <MessageSquare className="w-4 h-4 inline mr-2 text-violet-400" />{q}
                  </button>
                ))}
              </div>
            </div>
          )}
          {messages.map((msg, idx) => (
            <div key={idx} className={`flex gap-4 ${msg.role === 'user' ? 'justify-end' : ''}`}>
              {msg.role === 'assistant' && <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-violet-500 to-indigo-600 flex items-center justify-center flex-shrink-0"><Bot className="w-5 h-5 text-white" /></div>}
              <div className={`max-w-2xl ${msg.role === 'user' ? 'order-first' : ''}`}>
                <div className={`rounded-2xl px-4 py-3 ${msg.role === 'user' ? 'bg-gradient-to-r from-violet-600 to-indigo-600 text-white' : msg.error ? 'bg-red-900/30 border border-red-700/50 text-red-200' : 'bg-slate-800/80 border border-slate-700/50 text-slate-100'}`}>
                  <p className="whitespace-pre-wrap">{msg.content}</p>
                </div>
                {msg.sources && msg.sources.length > 0 && (
                  <div className="mt-3">
                    <button onClick={() => setExpandedSources(p => ({...p, [idx]: !p[idx]}))} className="flex items-center gap-2 text-sm text-slate-400 hover:text-violet-400">
                      {expandedSources[idx] ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
                      {msg.sources.length} source{msg.sources.length > 1 ? 's' : ''} • {msg.queryTime}ms
                    </button>
                    {expandedSources[idx] && (
                      <div className="mt-3 space-y-2">
                        {msg.sources.map((src, sIdx) => (
                          <div key={sIdx} className="rounded-xl bg-slate-800/50 border border-slate-700/30 p-3">
                            <div className="flex items-start justify-between gap-2">
                              <div><p className="font-medium text-white text-sm">{src.call_title}</p><p className="text-xs text-slate-400 mt-0.5">{src.call_date} {src.customer_company && `• ${src.customer_company}`}</p></div>
                              <span className="text-xs px-2 py-0.5 rounded-full bg-violet-500/20 text-violet-300 border border-violet-500/30">{(src.relevance_score * 100).toFixed(0)}%</span>
                            </div>
                            <p className="text-xs text-slate-400 mt-2 line-clamp-2">{src.excerpt}</p>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                )}
              </div>
              {msg.role === 'user' && <div className="w-8 h-8 rounded-lg bg-slate-700 flex items-center justify-center flex-shrink-0"><User className="w-5 h-5 text-slate-300" /></div>}
            </div>
          ))}
          {loading && (
            <div className="flex gap-4">
              <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-violet-500 to-indigo-600 flex items-center justify-center"><Bot className="w-5 h-5 text-white" /></div>
              <div className="rounded-2xl bg-slate-800/80 border border-slate-700/50 px-4 py-3">
                <div className="flex items-center gap-2 text-slate-400"><Loader2 className="w-4 h-4 animate-spin" /><span>Analyzing transcripts...</span></div>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>
      </main>
      <footer className="bg-slate-800/50 backdrop-blur-sm border-t border-slate-700/50 px-4 py-4">
        <div className="max-w-4xl mx-auto flex gap-3">
          <input type="text" value={input} onChange={e => setInput(e.target.value)} onKeyDown={e => e.key === 'Enter' && sendMessage(input)} placeholder="Ask about your sales calls..." disabled={loading} className="flex-1 bg-slate-900/50 border border-slate-700/50 rounded-xl px-4 py-3 text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-violet-500/50 disabled:opacity-50" />
          <button onClick={() => sendMessage(input)} disabled={!input.trim() || loading} className="px-5 py-3 bg-gradient-to-r from-violet-600 to-indigo-600 hover:from-violet-500 hover:to-indigo-500 disabled:from-slate-700 disabled:to-slate-700 text-white rounded-xl font-medium flex items-center gap-2 disabled:cursor-not-allowed">
            <Send className="w-5 h-5" />
          </button>
        </div>
      </footer>
    </div>
  );
}
