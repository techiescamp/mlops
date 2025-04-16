document.getElementById('prediction_form').addEventListener('submit', (e) => {
    e.preventDefault()
    const formData = new FormData(e.target)
    const data = {}
    formData.forEach((value, key) => {
        data[key] = value
    })
    let keys = ['Age', 'Years at Company', 'Monthly Income', 'Number of Promotions', 'Company Tenure', 'Number of Dependents']
    keys.forEach(item => data[item] = Number(data[item]))
    console.log(data)

    fetch('/predict', {
        method: 'POST',
        body: JSON.stringify(data),
        headers: { 'Content-Type': 'application/json' }
    })
    .then(res => res.json())
    .then(result => {
        console.log(result)
        const showResult = document.getElementById('result')
        showResult.style.display = 'block'
        showResult.innerHTML = `<b>Prediction:</b> ${result.prediction}`
    })
})