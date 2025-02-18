import React, { useContext, useState } from 'react'
import axios from 'axios'
import { LoadingContext } from '../App'

const apiurl = process.env.REACT_APP_BACKEND_URL

const Summarize = () => {
    const { loading, setLoading } = useContext(LoadingContext)
    const [text, setText] = useState('')
    const [summary, setSummary] = useState('')

    const handleSummarize = async (e) => {
        e.preventDefault()
        setLoading(true)
        try {
            const response = await axios.post(`${apiurl}/api/summarize`, { text } )
            if(response) setLoading(false)
            setSummary(response.data)
        } catch(err) {
            setSummary({message: err})
        }
    }

    const handleClear = () => {
        setSummary('')
        setText('')
    }

  return (
    <>
        <form onSubmit={handleSummarize}>
            <h2> Summarize Text</h2>
            <textarea 
                rows='10'
                cols='50'
                value={text}
                onChange={(e) => setText(e.target.value)}
            >
            </textarea>
            <div className='btns'>
                <button>Summarize</button>
                <button className='clear' onClick={handleClear}>Clear</button>
            </div>

        </form>

        <div className='result'>
            <h3>Summary</h3>
            {loading && <p className='px-3 py-1'>Loading....</p>}
            <p className='warning'>{summary.message && summary.message}</p>
            <p className='result'>{ summary.summary_text }</p>
        </div>
    </>
  )
}

export default Summarize