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

    fetch('/predict', {
        method: 'POST',
        body: JSON.stringify({ data: newData }),
        headers: { 'Content-Type': 'application/json' }
    })
    .then(res => res.json())
    .then(result => {
        console.log(result)
        const showResult = document.getElementById('result')
        showResult.style.display = 'block'
        showResult.innerHTML = `
            <b>Prediction Score:</b> ${result.prediction}
            <br/>
            <b>Risk Factor:</b> <span id="risk_level">${result.risk_level}</span>
            <br/>
            <b>Recommendation:</b> ${result.recommendation}
        `
        // Set color AFTER element is added
        const riskEl = document.getElementById('risk_level')
        if (result.risk_level === 'High') {
            riskEl.style.color = 'red'
        } else if (result.risk_level === 'Medium') {
            riskEl.style.color = 'orange'
        } else {
            riskEl.style.color = 'green'
        }
    })
})