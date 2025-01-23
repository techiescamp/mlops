import { Routes, Route, Link } from 'react-router-dom'
import './App.css'
import Home from './Home';
import Summarize from "./components/Summarize";
import SentimentAnalysis from './components/SentimentAnalysis';
import TextGeneration from './components/TextGeneration';
import QuestionAnswer from './components/QuestionAnswer';
import TableQA from './components/TableQA';
import { createContext, useState } from 'react';

export const LoadingContext = createContext()

function App() {
  const [loading, setLoading] = useState(false)

  return (
    <div className="App">
      <div id='header'>
        <h1>LLM Hugging Face Starter Projects</h1>
        <ul>
          <li><Link to='/' id='homeLink'>Home</Link></li>
        </ul>
      </div>

      <div className="tabBar">
        <ul>
          <li><Link to='/summarize'>Summarize</Link></li>
          <li><Link to='/sentiment-analysis'>Sentiment Analysis</Link></li>
          <li><Link to='/text-generation'>Text Generation</Link></li>
          <li><Link to='/question-answer'>Question Answer</Link></li>
          <li><Link to='/table-question-answer'>Table Question Answer</Link></li>
        </ul>
      </div>

      <div className="main">
        <LoadingContext.Provider value={{loading, setLoading}}>
          <Routes>
            <Route exact path='/' Component={Home} />
            <Route path='/summarize' Component={Summarize} />
            <Route path='/sentiment-analysis' Component={SentimentAnalysis} />
            <Route path='/text-generation' Component={TextGeneration} />
            <Route path='/question-answer' Component={QuestionAnswer} />
            <Route path='/table-question-answer' Component={TableQA} />
          </Routes>
        </LoadingContext.Provider>
      </div>

      <div id='copyright'>
        <p>&copy; techiescamp 2025</p>
      </div>
    </div>
  );
}

export default App;
