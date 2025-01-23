import React, { useContext, useState } from 'react'
import axios from 'axios'
import { LoadingContext } from '../App'

const QuestionAnswer = () => {
    const { loading, setLoading } = useContext(LoadingContext)

    const [input, setInput] = useState('')
    const [context, setContext] = useState('')
    const [result, setResult] = useState('')

    const handleQuestion = async(e) => {
        e.preventDefault()
        setLoading(true)
        try {
            const response = await axios.post('http://localhost:5000/api/question-answer', { input, context })
            if(response) setLoading(false)
            setResult(response.data)
        } catch(err) {
            setResult({message: `Error in sending API call to backend: ${err}`})
        }
    }   


  return (
    <>
        <form onSubmit={handleQuestion}>
            <h2>Question Answering</h2>
            <p>Your Context</p>
            <textarea
                className='qa'
                rows='4'
                cols='50'
                value={context}
                onChange={(e) => setContext(e.target.value)}
            >
            </textarea>

            <p>Your Question</p>
            <input
                type='text'
                value={input}
                onChange={(e) => setInput(e.target.value)}
            />
            <button>Generate</button>
        </form>

        <div className='result'>
            <h3>Answer to your question</h3>
            {loading && <p className='px-3 py-1'>Loading....</p>}

            <p>{result.message && result.message }</p>
            <p id='pos'>{result.answer}</p>
        </div>
    </>
  )
}

export default QuestionAnswer