import React, { useState, useEffect, useRef } from 'react';
import { 
  Upload, FileText, Send, Trash2, Play, CheckCircle2, 
  ChevronRight, ChevronDown, RefreshCw, Layers, Award, 
  Terminal, HelpCircle, X, ShieldAlert, BookOpen, AlertCircle
} from 'lucide-react';

export default function App() {
  // Tabs and general UI state
  const [activeTab, setActiveTab] = useState('chat');
  const [papers, setPapers] = useState([]);
  const [llmProvider, setLlmProvider] = useState('gemini');
  
  // Chat state
  const [messages, setMessages] = useState([]);
  const [inputText, setInputText] = useState('');
  const [isChatLoading, setIsChatLoading] = useState(false);
  const [activeCitation, setActiveCitation] = useState(null);
  
  // Upload state
  const [isUploading, setIsUploading] = useState(false);
  const [dragActive, setDragActive] = useState(false);
  const [uploadError, setUploadError] = useState(null);
  
  // Evaluation state
  const [evalResults, setEvalResults] = useState([]);
  const [isEvalLoading, setIsEvalLoading] = useState(false);
  const [runEvalMessage, setRunEvalMessage] = useState(null);

  // Refs
  const chatEndRef = useRef(null);
  const fileInputRef = useRef(null);

  // Auto-scroll chat to bottom
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isChatLoading]);

  // Load papers on startup
  useEffect(() => {
    fetchPapers();
    fetchEvalResults();
  }, []);

  const fetchPapers = async () => {
    try {
      const response = await fetch('/api/papers');
      if (response.ok) {
        const data = await response.json();
        setPapers(data);
      }
    } catch (error) {
      console.error('Error fetching papers:', error);
    }
  };

  const fetchEvalResults = async () => {
    try {
      const response = await fetch('/api/eval');
      if (response.ok) {
        const data = await response.json();
        setEvalResults(data);
      }
    } catch (error) {
      console.error('Error fetching evaluation results:', error);
    }
  };

  const handleDeletePaper = async (paperId, e) => {
    e.stopPropagation();
    if (!confirm(`Are you sure you want to delete paper '${paperId}'?`)) return;
    
    try {
      const response = await fetch(`/api/papers/${paperId}`, { method: 'DELETE' });
      if (response.ok) {
        fetchPapers();
        // Add a system notification to the chat
        setMessages(prev => [...prev, {
          sender: 'system',
          text: `Paper '${paperId}' deleted successfully.`
        }]);
      } else {
        alert('Failed to delete paper');
      }
    } catch (error) {
      console.error('Error deleting paper:', error);
    }
  };

  // Drag and Drop files
  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = async (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      uploadFile(e.dataTransfer.files[0]);
    }
  };

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      uploadFile(e.target.files[0]);
    }
  };

  const uploadFile = async (file) => {
    setIsUploading(true);
    setUploadError(null);
    
    const formData = new FormData();
    formData.append('file', file);
    
    try {
      const response = await fetch('/api/papers/upload', {
        method: 'POST',
        body: formData,
      });
      
      if (response.ok) {
        const newPaper = await response.json();
        fetchPapers();
        setMessages(prev => [...prev, {
          sender: 'system',
          text: `Successfully ingested and chunked paper: "${newPaper.title}" (${newPaper.chunk_count} chunks generated).`
        }]);
      } else {
        const errorData = await response.json();
        setUploadError(errorData.detail || 'Failed to process PDF.');
      }
    } catch (error) {
      setUploadError('Connection error while uploading paper.');
      console.error('Upload error:', error);
    } finally {
      setIsUploading(false);
      if (fileInputRef.current) fileInputRef.current.value = '';
    }
  };

  const handleSendMessage = async (e) => {
    e.preventDefault();
    if (!inputText.trim()) return;
    if (papers.length === 0) {
      alert('Please upload at least one research paper first.');
      return;
    }

    const userQuery = inputText;
    setInputText('');
    setMessages(prev => [...prev, { sender: 'user', text: userQuery }]);
    setIsChatLoading(true);

    try {
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: userQuery, provider: llmProvider })
      });

      if (response.ok) {
        const data = await response.json();
        setMessages(prev => [...prev, {
          sender: 'ai',
          text: data.answer,
          trace: data.trace,
          chunks: data.chunks
        }]);
      } else {
        const errorData = await response.json();
        setMessages(prev => [...prev, {
          sender: 'system',
          isError: true,
          text: errorData.detail || 'An error occurred during orchestration.'
        }]);
      }
    } catch (error) {
      setMessages(prev => [...prev, {
        sender: 'system',
        isError: true,
        text: 'Failed to connect to the backend server.'
      }]);
    } finally {
      setIsChatLoading(false);
    }
  };

  const runEvaluation = async () => {
    setIsEvalLoading(true);
    setRunEvalMessage('Evaluating chunking configs on 15 Q&A pairs (takes 5-10s)...');
    try {
      const response = await fetch('/api/eval/run', { method: 'POST' });
      if (response.ok) {
        const results = await response.json();
        setEvalResults(results);
        setRunEvalMessage('Evaluation run completed successfully!');
      } else {
        const errorData = await response.json();
        setRunEvalMessage(`Evaluation failed: ${errorData.detail}`);
      }
    } catch (error) {
      setRunEvalMessage('Failed to connect to evaluation API.');
    } finally {
      setIsEvalLoading(false);
    }
  };

  const toggleTrace = (index) => {
    setMessages(prev => prev.map((msg, idx) => {
      if (idx === index) {
        return { ...msg, showTrace: !msg.showTrace };
      }
      return msg;
    }));
  };

  return (
    <div className="app-container">
      {/* Sidebar */}
      <div className="sidebar">
        <div className="logo-section">
          <span className="logo-icon">✈️</span>
          <h1>PaperPilot</h1>
        </div>

        {/* Paper Ingestion */}
        <div className="upload-container">
          <div 
            className={`upload-zone ${dragActive ? 'drag-active' : ''}`}
            onDragEnter={handleDrag}
            onDragLeave={handleDrag}
            onDragOver={handleDrag}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
          >
            <Upload className="upload-icon" />
            <input 
              ref={fileInputRef} 
              type="file" 
              accept=".pdf" 
              onChange={handleFileChange} 
            />
            {isUploading ? (
              <div className="upload-text pulse-animation">
                Processing and chunking...
              </div>
            ) : (
              <div className="upload-text">
                <span>Click to upload</span> or drag and drop a PDF research paper
              </div>
            )}
          </div>
          {uploadError && (
            <div style={{ color: 'var(--accent)', fontSize: '12px', marginTop: '8px', display: 'flex', gap: '4px', alignItems: 'center' }}>
              <AlertCircle size={14} /> {uploadError}
            </div>
          )}
        </div>

        {/* Ingested Papers List */}
        <div className="papers-section">
          <h2 className="section-title">Ingested Papers ({papers.length})</h2>
          <div className="papers-list">
            {papers.length === 0 ? (
              <div style={{ color: 'var(--text-muted)', fontSize: '13px', textAlign: 'center', padding: '20px 0' }}>
                No papers uploaded yet.
              </div>
            ) : (
              papers.map((paper) => (
                <div key={paper.id} className="paper-card">
                  <div className="paper-header">
                    <div className="paper-title" title={paper.title}>{paper.title}</div>
                    <button className="delete-btn" onClick={(e) => handleDeletePaper(paper.id, e)}>
                      <Trash2 size={14} />
                    </button>
                  </div>
                  <div className="paper-meta">
                    <span><BookOpen size={11} /> {paper.page_count} pages</span>
                    <span><Layers size={11} /> {paper.chunk_count} chunks</span>
                  </div>
                </div>
              ))
            )}
          </div>
        </div>

        {/* LLM Provider Setup */}
        <div className="settings-section">
          <div className="provider-selector">
            <label className="provider-label">LLM ORCHESTRATOR</label>
            <div className="select-wrapper">
              <select 
                value={llmProvider} 
                onChange={(e) => setLlmProvider(e.target.value)}
              >
                <option value="gemini">Gemini (gemini-2.5-flash)</option>
                <option value="openai">OpenAI (gpt-4o-mini)</option>
                <option value="anthropic">Anthropic (claude-3-5-sonnet)</option>
              </select>
            </div>
          </div>
          <div style={{ fontSize: '11px', color: 'var(--text-muted)', lineHeight: '1.4' }}>
            Ensure the corresponding API key is loaded in your <code>.env</code> file.
          </div>
        </div>
      </div>

      {/* Main panel */}
      <div className="main-content">
        <header className="app-header">
          <div className="nav-tabs">
            <button 
              className={`tab-btn ${activeTab === 'chat' ? 'active' : ''}`}
              onClick={() => setActiveTab('chat')}
            >
              <Terminal size={16} /> Grounded Chat Q&A
            </button>
            <button 
              className={`tab-btn ${activeTab === 'eval' ? 'active' : ''}`}
              onClick={() => setActiveTab('eval')}
            >
              <Award size={16} /> Retrieval Evaluation
            </button>
          </div>
        </header>

        {activeTab === 'chat' ? (
          <div className="chat-panel">
            <div className="chat-history">
              {messages.length === 0 ? (
                <div className="empty-state">
                  <div className="empty-state-icon">✈️</div>
                  <h3>Welcome to PaperPilot</h3>
                  <p>
                    Ask cross-paper or single-paper comparison questions. 
                    The agent will decide which papers to query, run multiple search steps, 
                    and synthesize a grounded answer.
                  </p>
                  {papers.length === 0 && (
                    <div style={{ 
                      marginTop: '16px', 
                      padding: '10px 14px', 
                      background: 'rgba(99, 102, 241, 0.1)', 
                      borderRadius: '8px', 
                      fontSize: '12px',
                      color: 'var(--primary)',
                      border: '1px solid rgba(99, 102, 241, 0.2)',
                      display: 'flex',
                      alignItems: 'center',
                      gap: '8px'
                    }}>
                      <AlertCircle size={14} /> Start by uploading a PDF paper in the left sidebar!
                    </div>
                  )}
                </div>
              ) : (
                messages.map((msg, idx) => {
                  if (msg.sender === 'system') {
                    return (
                      <div key={idx} style={{ 
                        alignSelf: 'center', 
                        fontSize: '12px', 
                        color: msg.isError ? 'var(--accent)' : 'var(--secondary)', 
                        background: msg.isError ? 'rgba(236, 72, 153, 0.1)' : 'rgba(20, 184, 166, 0.1)',
                        border: `1px solid ${msg.isError ? 'rgba(236, 72, 153, 0.2)' : 'rgba(20, 184, 166, 0.2)'}`,
                        padding: '6px 16px',
                        borderRadius: '20px',
                        display: 'flex',
                        gap: '6px',
                        alignItems: 'center'
                      }}>
                        {msg.isError ? <ShieldAlert size={14} /> : <CheckCircle2 size={14} />}
                        {msg.text}
                      </div>
                    );
                  }

                  const isUser = msg.sender === 'user';
                  return (
                    <div key={idx} className={`message-wrapper ${isUser ? 'user' : 'ai'}`}>
                      <div className="message-avatar">
                        {isUser ? 'U' : 'AI'}
                      </div>
                      <div className="message-content">
                        <div className="message-bubble">
                          <p style={{ whiteSpace: 'pre-wrap' }}>{msg.text}</p>

                          {/* Render search trace logs */}
                          {!isUser && msg.trace && msg.trace.length > 0 && (
                            <div className="chat-trace-container">
                              <div className="trace-header" onClick={() => toggleTrace(idx)}>
                                {msg.showTrace ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
                                Agent Retrieval Logs ({msg.trace.length} turns)
                              </div>
                              {msg.showTrace && (
                                <div className="trace-steps">
                                  {msg.trace.map((t, tIdx) => (
                                    <div key={tIdx} className={`trace-step ${t.action === 'search' ? 'success' : ''}`}>
                                      {t.action === 'search' ? (
                                        <>
                                          <Play size={10} />
                                          <span>Searched {t.calls.map(c => `"${c.paper_id}" (${c.query})`).join(', ')}</span>
                                        </>
                                      ) : (
                                        <>
                                          <CheckCircle2 size={10} />
                                          <span>Generated response</span>
                                        </>
                                      )}
                                    </div>
                                  ))}
                                </div>
                              )}
                            </div>
                          )}
                        </div>

                        {/* Citation badges for AI message */}
                        {!isUser && msg.chunks && msg.chunks.length > 0 && (
                          <div className="citations-list">
                            <span style={{ fontSize: '11px', color: 'var(--text-muted)', display: 'flex', alignItems: 'center', marginRight: '4px' }}>
                              Grounding Sources:
                            </span>
                            {msg.chunks.map((chunk, cIdx) => (
                              <button 
                                key={cIdx} 
                                className="citation-badge"
                                onClick={() => setActiveCitation(chunk)}
                              >
                                <BookOpen size={10} />
                                {chunk.paper_id} (P. {chunk.primary_page}, Sec. {chunk.primary_section})
                              </button>
                            ))}
                          </div>
                        )}
                      </div>
                    </div>
                  );
                })
              )}
              {isChatLoading && (
                <div className="message-wrapper ai">
                  <div className="message-avatar">AI</div>
                  <div className="message-content">
                    <div className="message-bubble" style={{ background: 'var(--bubble-ai)', border: '1px solid var(--panel-border)' }}>
                      <div className="loading-dots">
                        <span></span>
                        <span></span>
                        <span></span>
                      </div>
                    </div>
                  </div>
                </div>
              )}
              <div ref={chatEndRef} />
            </div>

            {/* Input Bar */}
            <div className="chat-input-container">
              <form onSubmit={handleSendMessage} className="chat-form">
                <input 
                  type="text" 
                  value={inputText}
                  onChange={(e) => setInputText(e.target.value)}
                  placeholder={papers.length === 0 ? "Upload papers to begin chatting..." : "Compare paper methods, query results, ask cross-paper questions..."}
                  className="chat-input"
                  disabled={isChatLoading || papers.length === 0}
                />
                <button 
                  type="submit" 
                  className="send-btn" 
                  disabled={isChatLoading || !inputText.trim() || papers.length === 0}
                >
                  <Send size={16} />
                </button>
              </form>
            </div>
          </div>
        ) : (
          /* Evaluation Dashboard View */
          <div className="eval-panel">
            <div className="eval-header">
              <div className="eval-title">
                <h2>Retrieval Performance Evaluation</h2>
                <p>Verify search accuracy and compare chunking configurations using a dataset of 15 reference Q&A pairs.</p>
              </div>
              <button 
                className="run-eval-btn"
                onClick={runEvaluation}
                disabled={isEvalLoading}
              >
                {isEvalLoading ? <RefreshCw className="pulse-animation" size={16} /> : <Play size={16} />}
                Run Evaluation Suite
              </button>
            </div>

            {runEvalMessage && (
              <div style={{ 
                marginBottom: '20px', 
                padding: '12px 18px', 
                background: 'rgba(20, 184, 166, 0.1)', 
                border: '1px solid rgba(20, 184, 166, 0.2)',
                borderRadius: '8px',
                fontSize: '13px',
                color: 'var(--secondary)',
                display: 'flex',
                alignItems: 'center',
                gap: '8px'
              }}>
                <CheckCircle2 size={16} />
                {runEvalMessage}
              </div>
            )}

            <div className="eval-grid">
              <div className="eval-card">
                <h3 className="eval-card-title">Configuration Benchmarks</h3>
                <table className="eval-table">
                  <thead>
                    <tr>
                      <th>Configuration</th>
                      <th>Hit Rate @ Top-3</th>
                      <th>Hit Rate @ Top-5</th>
                    </tr>
                  </thead>
                  <tbody>
                    {evalResults.length === 0 ? (
                      <tr>
                        <td colSpan="3" style={{ textAlign: 'center', color: 'var(--text-muted)', padding: '30px' }}>
                          No evaluation results found. Click "Run Evaluation Suite" above to run benchmarks.
                        </td>
                      </tr>
                    ) : (
                      evalResults.map((res, idx) => (
                        <tr key={idx}>
                          <td style={{ fontWeight: 500 }}>{res.config}</td>
                          <td className="hit-rate-cell">{res.hit_rate_top3}%</td>
                          <td className="hit-rate-cell">{res.hit_rate_top5}%</td>
                        </tr>
                      ))
                    )}
                  </tbody>
                </table>
              </div>

              <div className="eval-card">
                <h3 className="eval-card-title">Benchmark Logic</h3>
                <div className="eval-summary-text">
                  <p>
                    <strong>Metric:</strong> The evaluation pipeline measures the <em>Hit Rate</em> across 15 Q&A pairs (5 queries per paper). 
                  </p>
                  <p>
                    <strong>Hit Definition:</strong> A query counts as a "Hit" if the ground-truth page (defined manually for each query) is correctly retrieved in the top-K returned chunks.
                  </p>
                  <p>
                    <strong>Goal:</strong> Optimize parameters (chunk size, overlap, etc.) to achieve maximum retrieval performance before querying the LLM, preventing model hallucination.
                  </p>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Citation Detail Sidebar Drawer */}
        {activeCitation && (
          <div className="citation-drawer">
            <div className="drawer-header">
              <h3>Source Grounding Details</h3>
              <button 
                className="close-drawer-btn" 
                onClick={() => setActiveCitation(null)}
              >
                <X size={16} />
              </button>
            </div>
            <div className="drawer-content">
              <div className="drawer-metadata">
                <div className="meta-row">
                  <span className="meta-label">Paper Name</span>
                  <span className="meta-val">{activeCitation.paper_title}</span>
                </div>
                <div className="meta-row">
                  <span className="meta-label">Source Page</span>
                  <span className="meta-val">Page {activeCitation.primary_page}</span>
                </div>
                <div className="meta-row">
                  <span className="meta-label">Section</span>
                  <span className="meta-val">{activeCitation.primary_section}</span>
                </div>
                <div className="meta-row">
                  <span className="meta-label">Vector Distance</span>
                  <span className="meta-val">{activeCitation.score.toFixed(4)}</span>
                </div>
              </div>
              <div style={{ fontSize: '11px', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '1px', fontWeight: 600 }}>
                Retrieved Chunk Text
              </div>
              <div className="chunk-text-box">
                {activeCitation.text}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
