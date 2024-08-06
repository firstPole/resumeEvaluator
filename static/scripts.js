$(document).ready(function() {
    var matchChartInstance = null;

    // Function to reset the form and UI elements
    function resetForm() {
        console.log('Resetting form...');
        $('#resume-form')[0].reset(); // Reset form fields
        $('#result').hide(); // Hide the result section
        $('#matchingTags').empty(); // Clear matching tags
        $('#non-matching-tags').empty(); // Clear non-matching tags
        $('#keywordDensity').empty(); // Clear keyword density
        $('#languageTone').empty(); // Clear language tone
        $('#emphasis').empty(); // Clear emphasis
        $('#socialResponsibility').empty(); // Clear social responsibility
        $('#sentiment-analysis-result').empty(); // Clear sentiment analysis
        $('#progress-bar').hide(); // Hide the progress bar
        $('#progress-bar-inner').css('width', '0%'); // Reset progress bar width
        $('#progress-text').text('0%'); // Reset progress text
        $('#matchText').empty(); // Clear match text
        $('#matchScoreText').empty(); // Clear match score text
        $('#improvementSuggestions').hide(); // Hide improvement suggestions

        // Destroy the existing Chart.js instance
        if (matchChartInstance) {
            matchChartInstance.destroy();
            matchChartInstance = null;
        }

        // Ensure canvas exists and is properly reset
        var $canvas = $('#matchChart');
        if ($canvas.length === 0) {
            console.error('Canvas element with id matchChart is not found.');
        } else {
            var canvas = $canvas[0];
            var ctx = canvas.getContext('2d');
            if (ctx) {
                ctx.clearRect(0, 0, canvas.width, canvas.height); // Clear canvas
            }
        }

        // Disable the reset button after reset
        $('#reset-button').prop('disabled', true);
    }

    $(document).ready(function() {
        $('#feedback-form').submit(function(e) {
            e.preventDefault();
            const feedback = $('#feedback-text').val();
            // Handle the feedback submission (e.g., via AJAX)
            alert('Thank you for your feedback: ' + feedback);
            $('#feedbackModal').modal('hide');
        });
    });

    function updateResetButtonState() {
        const resumeFile = $('#resume_file').val(); // Assuming file input has ID `resume_file`
        const jobDescription = $('#job_description').val(); // Assuming job description input has ID `job_description`
        $('#reset-button').prop('disabled', !resumeFile && !jobDescription);
    }

    $('#checkout-button').click(function() {
        $("#progress-bar").hide(); // Ensure progress bar is hidden
        console.log("inside dropin UI");
        window.location.href = '/dropin';
    });

    $('#resume-form').on('submit', function(event) {
        event.preventDefault();

        const progressBarContainer = document.getElementById('progress-bar');
        const progressBarInner = document.getElementById('progress-bar-inner');
        const progressText = document.getElementById('progress-text');

        progressBarContainer.style.display = 'block';

        // Simulate progress
        let progress = 0;
        const interval = setInterval(() => {
            if (progress < 100) {
                progress += 10;
                progressBarInner.style.width = progress + '%';
                progressText.textContent = progress + '%';
            } else {
                clearInterval(interval);
                progressText.textContent = 'Evaluation Complete!';
            }
        }, 200);

        var formData = new FormData(this);
        formData.append('resume_file', $('#resume_file')[0].files[0]);
        formData.append('job_description', $('#job_description').val());
        console.log('FormData:', formData); // Log FormData for debugging

        function updateExtractionResults(data) {
            console.log('Update Extraction Results:', data); // Log the data to check its structure
        
            // Check if data is defined and has the expected properties
            var email = data && data.email ? data.email : 'No Data Available';
            var phoneNumber = data && data.phone_number ? data.phone_number : 'No Data Available';
            var visaInfo = data && data.visa_info ? data.visa_info : 'No Data Available';
        
            $('#extracted-email').text(email);
            $('#extracted-phone').text(phoneNumber);
            $('#extracted-visa').text(visaInfo);
            $('#extraction-results').show();
        }

        $.ajax({
            url: '/evaluate',
            type: 'POST',
            data: formData,
            contentType: false,
            processData: false,
            success: function(response) {
                console.log('Response:', response); // Log the response for debugging
                
                // Update extraction results
                updateExtractionResults(response);

                $('#progress-bar').hide();
                $('#result').show();

                var matchScore = response.match_score;
                var matchingTags = response.relevant_tags;
                var nonMatchingTags = response.non_matching_tags;
                var keywordDensity = response.job_analysis.keyword_density;

                // Destroy existing Chart.js instance if it exists
                if (matchChartInstance) {
                    matchChartInstance.destroy();
                }

                // Ensure canvas is created and accessible
                var canvas = document.getElementById('matchChart');
                if (!canvas) {
                    console.error('Canvas element is not found.');
                    return;
                }

                var ctx = canvas.getContext('2d');
                if (!ctx) {
                    console.error('Cannot get context for the canvas.');
                    return;
                }

                // Function to get CSS variable value
                function getCSSVariable(name) {
                    return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
                }

                // Create a new chart
                matchChartInstance = new Chart(ctx, {
                    type: 'pie',
                    data: {
                        labels: ['Matching Skills', 'Non-Matching Skills'],
                        datasets: [{
                            data: [matchingTags.length, nonMatchingTags.length],
                            backgroundColor: [ 
                                getCSSVariable('--chart-color-matching'),
                                getCSSVariable('--chart-color-non-matching')
                            ], // Updated colors
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false, // Allow the chart to fill its container
                        plugins: {
                            legend: {
                                position: 'top',
                                align: 'start', // Align legends to start (horizontal)
                                labels: {
                                    boxWidth: 20, // Adjust box width as needed
                                    padding: 20 // Add padding between the legends
                                }
                            },
                            tooltip: {
                                callbacks: {
                                    label: function(tooltipItem) {
                                        var label = tooltipItem.label;
                                        var value = tooltipItem.raw;
                                        return label + ': ' + value;
                                    }
                                }
                            }
                        },
                        layout: {
                            padding: {
                                left: 20,
                                right: 20,
                                top: 20,
                                bottom: 20
                            }
                        }
                    }
                });

                console.log('Chart created:', matchChartInstance); // Log chart creation

                var matchText = matchScore > 90 ? 'Strong Match' :
                                matchScore >= 70 ? 'Good Match' : 
                                matchScore >= 50 ? 'Weak Match' : 'Very Weak Match';
                $('#matchText').text(matchText);
                $('#matchScoreText').text(matchScore + '%');

                // Display tags
                $('#matchingTags').empty();
                $('#non-matching-tags').empty();

                matchingTags.forEach(function(tag) {
                    $('#matchingTags').append('<span class="tag matching">' + tag + '</span>'); // Added 'matching' class
                });

                nonMatchingTags.forEach(function(tag) {
                    $('#non-matching-tags').append('<span class="tag non-matching">' + tag + '</span>');
                });

                // Display sentiment analysis
                if (response.sentiment && response.sentiment.analysis) {
                    let sentimentIcon;
                    switch (response.sentiment.analysis) {
                        case 'Positive':
                            sentimentIcon = '🟢'; // Green circle
                            break;
                        case 'Negative':
                            sentimentIcon = '🔴'; // Red circle
                            break;
                        case 'Neutral':
                            sentimentIcon = '🟡'; // Yellow circle
                            break;
                        default:
                            sentimentIcon = '⚪'; // White circle (default case)
                    }
                    $('#sentiment-analysis-result').html(
                        '<div class="sentiment-result">' +
                        '<span class="sentiment-icon">' + sentimentIcon + '</span> ' + response.sentiment.analysis +
                        '</div>'
                    );
                } else {
                    $('#sentiment-analysis-result').text('No Data Available');
                }

                // Display keyword density
                $('#keywordDensity').empty();
                if (keywordDensity.length > 0) {
                    keywordDensity.forEach(function(item) {
                        $('#keywordDensity').append('<div class="keyword-bar">' + item.keyword + ': ' + item.density + '</div>');
                    });
                } else {
                    $('#keywordDensity').append('No Data Available');
                }

                // Update job analysis sections
                $('#languageTone').text(response.job_analysis.language_tone || 'No Data Available');
                $('#emphasis').text(response.job_analysis.emphasis || 'No Data Available');
                $('#socialResponsibility').text(response.job_analysis.social_responsibility || 'No Data Available');

                // Display improvement suggestions if score is below 50%
                if (matchScore < 50) {
                    var suggestions = `
                        <h4>Improvement Suggestions:</h4>
                        <p>Your resume and the job description have a low match score. Here are some tips to improve it:</p>
                        <ul>
                            <li>Ensure your resume includes keywords from the job description.</li>
                            <li>Highlight relevant skills and experiences that align with the job requirements.</li>
                            <li>Customize your resume for each job application.</li>
                        </ul>
                    `;
                    $('#improvementSuggestions').html(suggestions).show();
                } else {
                    $('#improvementSuggestions').hide();
                }

                // Enable the reset button
                $('#reset-button').prop('disabled', false);
                updateResetButtonState();
            },
            error: function(xhr, status, error) {
                console.error('Error:', error);
                $('#progress-bar').hide();
                alert('An error occurred. Please try again.');
            }
        });
    });

    // Reset button event handler
    $('#reset-button').click(function() {
        resetForm();
    });

    // Update reset button state on form change
    $('#resume_file, #job_description').on('change keyup', function() {
        updateResetButtonState();
    });

    // Handle showing the modal and form reset
    $('#show-feedback-modal').click(function() {
        $('#feedbackModal').modal('show');
    });

    $('#feedbackModal').on('hidden.bs.modal', function () {
        resetForm(); // Reset the form when modal is closed
    });
});
