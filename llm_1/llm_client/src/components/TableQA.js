import React, { useContext, useState } from 'react'
import axios from 'axios'
import { LoadingContext } from '../App'

const apiurl = process.env.REACT_APP_BACKEND_URL

const TableQA = () => {
    const { loading, setLoading } = useContext(LoadingContext)

    const [ques, setQues] = useState('')
    const [file, setFile] = useState(null)
    const [answer, setAnswer] = useState('')

    const handleFileChange = (e) => {
        setFile(e.target.files[0])
    }

    const handleTableQA = async (e) => {
        e.preventDefault()
        setLoading(true)
        // const tableData = JSON.parse(table)
        if(!file) {
            alert("Please upload a file")
            return
        }
        const formData = new FormData()
        formData.append('file', file)
        formData.append('ques', ques)
        try {
            const response = await axios.post(`${apiurl}/api/table-question-answer`, formData, {
                headers: {'Content-Type': 'multipart/form-data' },
            })
            if(response) setLoading(false)
            setAnswer(response.data.answer)
        } catch(err) {
            console.error(`Error: ${err.message}`)
        }
    }

  return (
    <>
        <form onSubmit={handleTableQA}>
            <h2>Table Question Answer</h2>
            <p>Your question</p>
            <textarea
                className='qa'
                rows='3'
                cols='50'
                value={ques}
                onChange={(e) => setQues(e.target.value)}
            >
            </textarea>
            <p><small>Accept only .csv file</small></p>
            <input 
                type='file'
                onChange={handleFileChange}
            />
            <button>Get Answer</button>
        </form>

        <div className='result'>
            <h3>Answer ?</h3>
            {loading && <p className='px-3 py-1'>Loading....</p>}

            <p>{answer}</p>     
        </div>
           
    </>
  )
}

export default TableQA