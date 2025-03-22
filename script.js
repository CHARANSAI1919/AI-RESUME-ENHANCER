document.getElementById("enhanceBtn").addEventListener("click", async () => {
    const resumeFile = document.getElementById("resume").files[0];
    const jobDescription = document.getElementById("jobDescription").value;

    if (!resumeFile) {
        alert("Please upload a resume.");
        return;
    }

    // Show loading state
    const enhanceBtn = document.getElementById("enhanceBtn");
    enhanceBtn.disabled = true;
    enhanceBtn.textContent = "Processing...";

    const formData = new FormData();
    formData.append("resume", resumeFile);
    formData.append("job_description", jobDescription);

    try {
        const response = await fetch("/enhance", {
            method: "POST",
            body: formData,
        });

        if (!response.ok) {
            throw new Error("Failed to process resume.");
        }

        const result = await response.json();

        // Display corrected resume
        document.getElementById("correctedText").textContent = result.corrected_text;

        // Display missing keywords
        document.getElementById("missingKeywords").innerHTML = result.missing_keywords
            .map(keyword => `<li class="list-group-item">${keyword}</li>`)
            .join("");

        // Display resume score and feedback
        const scoreBreakdown = result.score_breakdown;
        document.getElementById("resumeScore").textContent = `Resume Score: ${scoreBreakdown.total_score}/100`;
        document.getElementById("resumeFeedback").textContent = scoreBreakdown.feedback;
        document.getElementById("scoreBreakdown").innerHTML = `
            <li class="list-group-item">Grammar: ${scoreBreakdown.grammar_score}/40</li>
            <li class="list-group-item">Keywords: ${scoreBreakdown.keyword_score}/30</li>
            <li class="list-group-item">Formatting: ${scoreBreakdown.formatting_score}/20</li>
            <li class="list-group-item">Structure: ${scoreBreakdown.structure_score}/10</li>
        `;

        // Display action verb suggestions
        document.getElementById("actionVerbSuggestions").innerHTML = result.action_verb_suggestions
            .map(suggestion => `<li class="list-group-item">${suggestion}</li>`)
            .join("");

        // Set download links
        document.getElementById("downloadLinkTxt").href = result.download_links.txt;
        document.getElementById("downloadLinkDocx").href = result.download_links.docx;
        document.getElementById("downloadLinkPdf").href = result.download_links.pdf;

        // Show results section
        document.getElementById("results").style.display = "block";
    } catch (error) {
        alert("An error occurred. Please try again.");
        console.error(error);
    } finally {
        // Reset button
        enhanceBtn.disabled = false;
        enhanceBtn.textContent = "Enhance Resume";
    }
});