const express = require('express')
const cors = require('cors')
const bodyParser = require('body-parser')
require('dotenv').config()
// route for HF
const { HfInference } = require('@huggingface/inference')
// table qa parsers
const multer = require('multer')
const csv = require('csv-parser')
// const pdfParse = require('pdf-parse')
const PDFParser = require('pdf2json')
const fs = require('fs')
const path = require('path')


const PORT = process.env.PORT;
const hf_key = String(process.env.HF_API_KEY);
// multer setup
const upload = multer({dest: path.join(__dirname, 'uploads')})

const app = express()
// hf
const hf = new HfInference(hf_key)

app.use(cors())
app.use(bodyParser.json())

// create route for hugging face api
app.post("/api/summarize", async (req, res) => {
    const { text } = req.body;
    try {
        const response = await hf.summarization({
            model: 'facebook/bart-large-cnn',
            inputs: text,
        })
        res.status(200).json(response)
        // output: {
        //    summary_text: '..........'
        //  }
    } catch(err) {
        res.status(500).json({
            message: `Error in summarizing text at backend ${err}`, 
        })
    }
})

// route for text generation 
app.post('/api/text-generation', async(req, res) => {
    const { textGen } = req.body
    console.log(textGen)
    try {
        const response = await hf.textGeneration({
            model: 'gpt2',
            inputs: textGen,
            
        })
        console.log(response)
        res.status(200).send(response)
        // output: {
        //    generated_text: '..........'
        //  }
        // openai-community/gpt
        // mistralai/Mistral-7B-Instruct-v0.3
    } catch(err) {
        res.status(500).json({
            message: `Error in generating text at backend ${err}`
        })
    }
})

// ---------------------------------------------------------------------------------------------
// route for updated text-generation with parameters
app.post('/api/update/text-generation', async (req, res) => {
    const { textGen } = req.body;
    // console.log('Input for text generation:', textGen);
    // EleutherAI/gpt-neo-2.7B
    try {
        const response = await hf.textGeneration({
            model: 'EleutherAI/gpt-neo-2.7B', // Use a larger, more diverse model if available
            inputs: textGen,
            parameters: {
                max_new_tokens: 50,
                temperature: 0.7,
                top_p: 0.9,
                top_k: 50,
                repetition_penalty: 1,
                seed: 42
            }
        });
        // const generatedText = response?.generated_text || 'No text generated';
        console.log('Generated text response:', response);

        // Optional post-processing
        const finalText = removeRepetitions(response.generated_text);

        res.status(200).send({ response: finalText });
    } catch (err) {
        console.error('Error in text generation:', err.message);
        res.status(500).json({
            message: `Error in generating text at backend: ${err.message}`
        });
    }
});

// Utility to remove repetitions
function removeRepetitions(text) {
    const sentences = text.split('. ');
    const uniqueSentences = [...new Set(sentences)];
    return uniqueSentences.join('. ');
}

// --------------------------------------------------------------------------------

// sentiment analysis or text classification
app.post('/api/sentiment-analysis', async(req, res) => {
    const { input } = req.body
    console.log(input)
    try {
        const response = await hf.textClassification({
            model: 'Liusuthu/my_text_classification_model_based_on_distilbert',
            inputs: input
        })
        res.status(200).send(response)
        // base-model: 'distilbert-base-uncased-finetuned-sst-2-english',
        // output: 
        // [
        //     {
        //         "label": 'POSITIVE',
        //         "score": 0.93
        //     },
        //     {
        //         "label": "NEGATIVE",
        //         "score": 0.245
        //     }
        // ]
    } catch(err) {
        res.status(500).json({
            message: `Error in classifying text at backednd: ${err}`
        })
    }
})

// create question answer API route
app.post('/api/question-answer', async(req, res) => {
    const { input, context } = req.body
    try {
        const response = await hf.questionAnswering({
            model: 'distilbert/distilbert-base-cased-distilled-squad',
            inputs: {
                question: input,
                context: context
            }
        })
        console.log(response)
        res.status(200).json(response)
        // { score: 0.972388505935669, start: 39, end: 46, answer: 'biryani' }
    } catch(err) {
        res.status(500).json({
            message: `Error in backend: ${err}`
        })
    }
})

// table qa
app.post('/api/table-question-answer', upload.single('file'), async(req, res) => {
    const { ques } = req.body
    const file = req.file

    if (!file || !fs.existsSync(file.path)) {
        return res.status(400).send('File not found or inaccessible.');
    }

    let table = []
    // determine file type
    if(file.mimetype === 'text/csv') {
        // parse csv file
        fs.createReadStream(file.path)
            .on('error', (error) => {
                console.error('Error reading file: ', error)
                res.status(500).json({message: `Error reading file: ${error.message}`})
            })
            .pipe(csv())
            .on('data', (row) => table.push(row))
            .on('end', async() => {
                console.log('end part reached')
                try {
                    const response = await hf.tableQuestionAnswering({
                        model: 'google/tapas-large-finetuned-wtq',
                        inputs: {
                            query: ques,
                            table: table
                        }
                    });
                    
                    // Clean up the uploaded file
                    fs.unlink(file.path, (err) => {
                        if (err) console.error('Error deleting file:', err);
                    });
                    console.log(response)
                    res.status(200).json(response);
                } catch (error) {
                    console.error('Error processing query:', error);
                    res.status(500).json({ message: `Error processing query: ${error.message}` });
                }
            })
            .on('error', (error) => {
                console.error('Error parsing CSV:', error);
                res.status(500).json({ message: `Error parsing CSV: ${error.message}` });
            });
    } 
})

// output
//  {
//     answer: 'Glass / Ceramics / Concrete',
//     coordinates: [ [ 1, 7 ] ],
//     cells: [ 'Glass / Ceramics / Concrete' ],
//     aggregator: 'NONE'
//   }

app.post('/api/doc-question-answer', upload.single('file'), async (req, res) => {
    const { ques } = req.body;
    const file = req.file;

    if (!file || !fs.existsSync(file.path)) {
        return res.status(400).send('File not found or inaccessible.');
    }

    if (file.mimetype === 'application/pdf') {
        console.log('Processing PDF file...');

        const pdfParser = new PDFParser(null, 1);

        pdfParser.on("pdfParser_dataError", (err) => {
            console.error("Error parsing PDF:", err);
            res.status(500).json({ message: `Error parsing PDF: ${err.message}` });
        });

        pdfParser.on("pdfParser_dataReady", async (pdfData) => {
            // console.log("Full PDF Data: ", JSON.stringify(pdfData, null, 2));
            try {
                // Extract the text content from the parsed PDF
                let pdfText = ""
                if(pdfData && pdfData.Pages) {
                    pdfText = pdfData.Pages.map(page => {
                        if(!page.Texts) return ""
                        return page.Texts.map(text => {
                            if(text.R && text.R.length > 0 && text.R[0].T) {
                                return decodeURIComponent(text.R[0].T)
                            }
                            return ""
                        }).join(" ")
                    }).join(" ").trim()
                } 
                console.log("Extracted PDF Content:", pdfText);
                if(!pdfText) {
                    pdfText = "No text content found in PDF"
                }

                // Call the Hugging Face Question Answering API
                const response = await hf.questionAnswering({
                    model: "deepset/roberta-base-squad2",
                    inputs: {
                        'question': ques,
                        'context': pdfText
                    }
                });

                // Clean up the uploaded file
                fs.unlink(file.path, (err) => {
                    if (err) console.error("Error deleting file:", err);
                });

                console.log("Response from Hugging Face API:", response);
                res.status(200).json({ output: response, content: pdfText });
            } catch (err) {
                console.error("Error during Hugging Face API call:", err);
                res.status(500).json({ message: `Error in backend: ${err.message}` });
            }
        });

        // Start parsing the uploaded PDF
        pdfParser.loadPDF(file.path);
    } else {
        res.status(400).send('Only PDF files are supported.');
    }
});


app.listen(PORT, () => {
    console.log(`Server is running at ${PORT}`)
});














// const mappedData = {}
//     if(table.length > 0) {
//         const headers = Object.keys(table[0])
//         console.log('headers: ', headers)
//         headers.forEach(head => {
//             mappedData[head] = table.map(row => String(row[head]))
//         })
//     }
//     console.log('map data: ', JSON.stringify(mappedData))
