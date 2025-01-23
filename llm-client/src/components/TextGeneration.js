import React, { useContext, useState } from 'react'
import axios from 'axios'
import { LoadingContext } from '../App'

const TextGeneration = () => {
    const { loading, setLoading } = useContext(LoadingContext)

    const [textGen, setTextGen] = useState({
        ques: '',
        isGenerated: false
    })
    const [result, setResult] = useState('')

    const handleGenerate = async(e) => {
        e.preventDefault()
        setLoading(true)
        try {
            // const response = await axios.post('http://localhost:5000/api/text-generation', { textGen: textGen.ques })
            const response = await axios.post('http://localhost:5000/api/update/text-generation', { textGen: textGen.ques })
            if(response) setLoading(false)
            setTextGen({
                ...textGen, 
                isGenerated: true
            })
            // setResult(response.data)
            setResult(response.data.response)
        }catch(err) {
            setResult(err)
        }
    }

  return (
    <div className='textGen-container'>
        <h2>Text Generation</h2>
        <p>{ result.message && result.message }</p>
        {loading && <p className='px-3 py-1'>Loading....</p>}

        <div className='textGen-window'>
            <p className='yourQues'>{textGen.isGenerated && textGen.ques}</p>
            {/* <p className='yourResult'>{ result.generated_text && result.generated_text }</p> */}
            <p className='yourResult'>{ result && result }</p>
        </div>

        <input
            type='text'
            value={textGen.ques}
            onChange={(e) => setTextGen({ ...textGen, ques: e.target.value })}
        />
        <button onClick={handleGenerate}>Generate</button>

    </div>
  )
}

export default TextGeneration