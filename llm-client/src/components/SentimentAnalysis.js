import React, { useEffect, useState, useContext } from 'react'
import axios from 'axios'
import { LoadingContext } from '../App'

const SentimentAnalysis = () => {
  const { loading, setLoading } = useContext(LoadingContext)
  
  const [input, setInput] = useState('')
  const [result, setResult] = useState('')
  const [analysis, setAnalysis] = useState(null)
  const [progressbar, setProgressbar] = useState({
    positive: 0,
    negative: 0
  })

  const handleAnalyze = async (e) => {
    e.preventDefault()
    setLoading(true)
    try {
      const response = await axios.post('http://localhost:5000/api/sentiment-analysis', { input })
      if(response)  setLoading(false)
      setResult(response.data)
    } catch (err) {
      setResult({ message: 'Error occured in calling API' })
    }
  }

  useEffect(() => {
    if (result) {
      const sentiment = result.filter(t => t.score > 0.5)
      setAnalysis(sentiment)

      // handle progress bar for positives
      const posLabel = result.find(t => t.label === 'POSITIVE')
      const negLabel = result.find(t => t.label === 'NEGATIVE')

      const posScore = posLabel ? (posLabel.score * 100).toFixed(2) : 0;
      const negScore = negLabel ? (negLabel.score * 100).toFixed(2) : 0;
      setProgressbar({
        positive: `${posScore}%`,
        negative: `${negScore}%`
      })

      function updatedProgressbar(elementId, score) {
        let progressWidth = 0;
        const element = document.getElementById(elementId)
        const id = setInterval(() => {
          if (progressWidth >= score) clearInterval(id)
          else {
            progressWidth++
            element.style.width = `${progressWidth}%`
            element.style.backgroundColor = score > 50 ? "#4caf50" : "#f44336"
          }
        }, 10)
      }
      updatedProgressbar('positive-progress', posScore)
      updatedProgressbar('negative-progress', negScore)

    }
  }, [result])

  return (
    <>
      <form onSubmit={handleAnalyze}>
        <h2>Sentiment Analysis</h2>
        <textarea
          rows='10'
          cols='50'
          value={input}
          onChange={(e) => setInput(e.target.value)}
        >
        </textarea>
        <button>Analyze</button>
      </form>

      <div className='result'>
        <h3>Analysis</h3>
        <p className='warning'>{result.message && result.message}</p>
        {loading && <p className='px-3 py-1'>Loading....</p>}

        {analysis && analysis.length > 0 && (
          analysis[0].label === 'POSITIVE' ? <p id='pos'>{analysis[0].label} &#128525;</p>
            : analysis[0].label === 'NEGATIVE' ? <p id='neg'>{analysis[0].label} &#128530;</p>
              : null
        )}

        <h3>Analyzer</h3>
        <div className='pg-container'>
          <h4>Positive</h4>
          <div id='progressbar-container'>
            <div id='positive-progress' className='progressbar'></div>
          </div>
          <p>{progressbar && progressbar.positive}</p>
        </div>
        
        <div className='pg-container'>
          <h4>Negative </h4>
          <div id='progressbar-container'>
            <div id='negative-progress' className='progressbar'></div>
          </div>
          <p>{progressbar && progressbar.negative}</p>
        </div>
                
      </div>
    </>
  )
}

export default SentimentAnalysis