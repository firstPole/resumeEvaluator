
$(document).ready(function() {
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
        $.ajax({
            url: '/evaluate',
            type: 'POST',
            data: formData,
            contentType: false,
            processData: false,
            success: function(response) {
                $('#progress-bar').hide();
                $('#result').show();

                var matchScore = response.match_score;
                var matchingTags = response.relevant_tags;
                var nonMatchingTags = response.non_matching_tags;
                var keywordDensity = response.job_analysis.keyword_density;

                // Display match score and chart
                var ctx = document.getElementById('matchChart').getContext('2d');
                var matchChart = new Chart(ctx, {
                    type: 'pie',
                    data: {
                        labels: ['Matching Skills', 'Non-Matching Skills'],
                        datasets: [{
                            data: [matchingTags.length, nonMatchingTags.length],
                            backgroundColor: ['#002244', '#b0bec5'],
                        }]
                    },
                    options: {
                        responsive: true,
                        plugins: {
                            legend: {
                                position: 'top',
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
                        }
                    }
                });

                var matchText = matchScore > 90 ? 'Strong Match' :
                                matchScore >= 70 ? 'Good Match' : 'Weak Match';
                $('#matchText').text(matchText);
                $('#matchScoreText').text(matchScore + '%');

                // Display tags
                $('#matchingTags').empty();
                $('#non-matching-tags').empty();

                matchingTags.forEach(function(tag) {
                    $('#matchingTags').append('<span class="tag">' + tag + '</span>');
                });

                nonMatchingTags.forEach(function(tag) {
                    $('#non-matching-tags').append('<span class="tag non-matching">' + tag + '</span>');
                });

                // Check if response.sentiment is defined and has the 'analysis' property
               // Check if response.sentiment is defined and has the 'analysis' property
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
                console.error('Sentiment analysis data is missing or undefined.');
            }

                // Display keyword density
                $('#keywordDensity').empty();
                keywordDensity.forEach(function(item) {
                    
                    $('#keywordDensity').append('<div class="keyword-bar" style="display: inline-block">' + item.keyword + ': ' + item.density + '</div>');
                });

                // Update job analysis sections
                $('#languageTone').text(response.job_analysis.language_tone);
                $('#emphasis').text(response.job_analysis.emphasis);
                $('#socialResponsibility').text(response.job_analysis.social_responsibility);
            }
        });
    });
});