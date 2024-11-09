const express = require('express');
const bodyParser = require('body-parser');
const { spawn } = require('child_process');  // For calling Python scripts

const app = express();
app.use(bodyParser.json());

// Endpoint to get recommendations
app.post('/api/getRecommendations', (req, res) => {
    const { location, budget, duration, ageGroups, category } = req.body;

    // Call the Python recommendation script
    const pythonProcess = spawn('python', ['recommendation_script.py', location, budget, duration, JSON.stringify(ageGroups), category]);

    pythonProcess.stdout.on('data', (data) => {
        const recommendations = JSON.parse(data.toString());
        res.json(recommendations);
    });

    pythonProcess.stderr.on('data', (data) => {
        console.error(`Error: ${data}`);
        res.status(500).send("Error generating recommendations");
    });
});

const PORT = process.env.PORT || 5000;
app.listen(PORT, () => console.log(`Server running on port ${PORT}`));
