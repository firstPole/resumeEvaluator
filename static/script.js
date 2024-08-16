$(document).ready(function() {
    var matchChartInstance = null;

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

        if (matchChartInstance) {
            matchChartInstance.destroy();
            matchChartInstance = null;
        }

        var $canvas = $('#matchChart');
        if ($canvas.length > 0) {
            var canvas = $canvas[0];
            var ctx = canvas.getContext('2d');
            if (ctx) {
                ctx.clearRect(0, 0, canvas.width, canvas.height); // Clear canvas
            }
        }

        $('#resultModal').modal('hide'); // Hide the modal dialog
        $('#spinner-overlay').hide(); // Hide the spinner
        $('#reset-button').prop('disabled', true);
    }

    function updateResetButtonState() {
        const resumeFile = $('#resume_file').val();
        const jobDescription = $('#job_description').val();
        $('#reset-button').prop('disabled', !resumeFile && !jobDescription);
    }

    $('#resume-form').on('submit', function(event) {
        event.preventDefault();

        $('#spinner-overlay').show(); // Show the spinner

        // const progressBarContainer = document.getElementById('progress-bar');
        // const progressBarInner = document.getElementById('progress-bar-inner');
        // const progressText = document.getElementById('progress-text');

        // progressBarContainer.style.display = 'block';

        // let progress = 0;
        // const interval = setInterval(() => {
        //     if (progress < 100) {
        //         progress += 10;
        //         progressBarInner.style.width = progress + '%';
        //         progressText.textContent = progress + '%';
        //     } else {
        //         clearInterval(interval);
        //         progressText.textContent = 'Evaluation Complete!';
        //     }
        // }, 200);

        var formData = new FormData(this);
        formData.append('resume_file', $('#resume_file')[0].files[0]);
        formData.append('job_description', $('#job_description').val());

        function updateExtractionResults(data) {
            console.log('Update Extraction Results:', data);
            var email = data && data.email ? data.email : 'Not Found';
            var phoneNumber = data && data.phone_number ? data.phone_number : 'Not Found';
            var visaInfo = data && data.visa_info ? data.visa_info : 'Not Found';
        
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
                console.log('Response:', response);

                $('#spinner-overlay').hide(); // Hide the spinner
                $('#progress-bar').hide(); // Hide the progress bar
                $('#resultModal').modal('show'); // Show the modal dialog

                updateExtractionResults(response);

                var matchScore = response.match_score;
                var matchingTags = response.relevant_tags;
                var nonMatchingTags = response.non_matching_tags;
                var keywordDensity = response.job_analysis.keyword_density;


                
                if (matchChartInstance) {
                    matchChartInstance.destroy();
                }

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

                function getCSSVariable(name) {
                    return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
                }

                matchChartInstance = new Chart(ctx, {
                    type: 'pie',
                    data: {
                        labels: ['Matching Skills', 'Non-Matching Skills'],
                        datasets: [{
                            data: [matchingTags.length, nonMatchingTags.length],
                            backgroundColor: [ 
                                getCSSVariable('--chart-color-matching'),
                                getCSSVariable('--chart-color-non-matching')
                            ],
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: {
                            legend: {
                                position: 'top',
                                align: 'start',
                                labels: {
                                    boxWidth: 20,
                                    padding: 20
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

                console.log('Chart created:', matchChartInstance);

                var matchText = matchScore > 90 ? 'Strong Match' :
                                matchScore >= 70 ? 'Good Match' : 
                                matchScore >= 50 ? 'Weak Match' : 'Very Weak Match';
                $('#matchText').text(matchText);
                var roundedScore = Math.round(matchScore);
                $('#matchScoreText').text(roundedScore + '');

                $('#matchingTags').empty();
                $('#non-matching-tags').empty();

                matchingTags.forEach(function(tag) {
                    $('#matchingTags').append('<span class="tag matching">' + tag + '</span>');
                });

                nonMatchingTags.forEach(function(tag) {
                    $('#non-matching-tags').append('<span class="tag non-matching">' + tag + '</span>');
                });

                $('#sentiment-analysis-result').empty();
                if (response.sentiment && response.sentiment.analysis) {
                    let sentimentIcon;
                    switch (response.sentiment.analysis) {
                        case 'Positive':
                            sentimentIcon = '🟢';
                            break;
                        case 'Negative':
                            sentimentIcon = '🔴';
                            break;
                        case 'Neutral':
                            sentimentIcon = '🟡';
                            break;
                        default:
                            sentimentIcon = '⚪';
                    }
                    $('#sentiment-analysis-result').html(
                        '<div class="sentiment-result">' +
                        '<span class="sentiment-icon">' + sentimentIcon + '</span> ' + response.sentiment.analysis +
                        '</div>'
                    );
                } else {
                    $('#sentiment-analysis-result').text('Not Found');
                }

                $('#keywordDensity').empty();
                if (keywordDensity.length > 0) {
                    keywordDensity.forEach(function(item) {
                        $('#keywordDensity').append('<div class="keyword-bar">' + item.keyword + ': ' + item.density + '</div>');
                    });
                } else {
                    $('#keywordDensity').text('Not Found');
                }

                $('#languageTone').text(response.job_analysis.language_tone || 'Not Found');
                $('#emphasis').text(response.job_analysis.emphasis || 'Not Found');
                $('#socialResponsibility').text(response.job_analysis.social_responsibility || 'Not Found');
                $('#improvementSuggestions').text(response.improvement_suggestions || 'No Suggestions Available');

                $('#downloadResumeBtn').attr('href', response.updated_resume_url);

            },
            error: function(xhr, status, error) {
                console.error('AJAX Error:', status, error);
                $('#spinner-overlay').hide(); // Hide the spinner
                $('#progress-bar').hide(); // Hide the progress bar
                $('#error-message').text('An error occurred. Please try again.').show();
                $('#reset-button').prop('disabled', false);
            }
        });
    });

    $('#reset-button').on('click', function() {
        resetForm();
    });

    $('#resume_file, #job_description').on('change keyup', function() {
        updateResetButtonState();
    });
    $('#checkout-button').click(function() {
        $("#progress-bar").hide(); // Ensure progress bar is hidden
        console.log("inside dropin UI");
        window.location.href = '/dropin';
    });
});
