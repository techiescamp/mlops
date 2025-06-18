document.getElementById('prediction_form').addEventListener('submit', (e) => {
    e.preventDefault()
    const formData = new FormData(e.target)
    const data = {}
    formData.forEach((value, key) => {
        data[key] = value
    })
    let keys = ['Age', 'Years at Company', 'Monthly Income', 'Number of Promotions', 'Company Tenure', 'Number of Dependents']
    keys.forEach(item => data[item] = Number(data[item]))
    const newData = {...data, 'employee_id': 8410}
    // console.log(newData)
    // console.log(PREDICTION_API_URL)

    fetch(PREDICTION_API_URL, {
        method: 'POST',
        body: JSON.stringify({ data: newData }),
        headers: { 'Content-Type': 'application/json' }
    })
    .then(res => res.json())
    .then(result => {
        console.log(result)
        let response;
        if(result.prediction === 1) {
            response = "Left"
        } else if(result.prediction === 0) {
            response = "Stayed" 
        } else {
            response = "undefined"
        }
        const showResult = document.getElementById('result')
        showResult.style.display = 'block'
        showResult.innerHTML = `<b>Prediction:</b> ${response}`
    })
})