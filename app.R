# Load required libraries
library(shiny)
library(dplyr)
library(tidymodels)
library(ranger)
library(vip)

# Load XGBoost model
load("rf_model_tidym.Rdata")


# Define UI for Shiny app
ui <- fluidPage(
  # App title
  titlePanel("Predict Heart Failure with Risk Factors"),
  # Sidebar layout with input and output definitions
  sidebarLayout(
    # Sidebar panel for input parameters
    sidebarPanel(
      # Input for age
      sliderInput(inputId = "Age", label = "Age (years)",
                  value = 50, min = 10, max = 100),
      # Input for sex
      selectInput(inputId = "Sex", label = "Sex",
                  choices = c("M", "F"), selected = "M"),
      # Input for chest pain type
      selectInput(inputId = "ChestPainType", label = "Chest Pain Type",
                  choices = c("ASY", "ATA", "NAP", "TA"), selected = "TA"),
      # Input for resting blood pressure
      sliderInput(inputId = "RestingBP", label = "Resting Blood Pressure (mm Hg)",
                  value = 130, min = 0, max = 250),
      # Input for serum cholesterol
      sliderInput(inputId = "Cholesterol", label = "Serum Cholesterol (mg/dl)",
                  value = 250, min = 0, max = 650),
      # Input for fasting blood sugar
      selectInput(inputId = "FastingBS", label = "Fasting Blood Sugar > 120 mg/dl",
                  choices = c("N", "Y"), selected = "N"),
      # Input for resting electrocardiographic results
      selectInput(inputId = "RestingECG", label = "Resting Electrocardiographic Results",
                  choices = c("Normal", "ST", "LVH"), selected = "Normal"),
      # Input for maximum heart rate achieved
      sliderInput(inputId = "MaxHR", label = "Maximum Heart Rate Achieved",
                  value = 150, min = 50, max = 250),
      # Input for exercise-induced angina
      selectInput(inputId = "ExerciseAngina", label = "Exercise-Induced Angina",
                  choices = c("Y", "N"), selected = "N"),
      # Input for ST depression induced by exercise relative to rest
      sliderInput(inputId = "Oldpeak", label = "ST Depression Induced by Exercise Relative to Rest",
                  value = 1.5, min = 0, max = 6, step=0.5),
      # Input for the slope of the peak exercise ST segment
      selectInput(inputId = "ST_Slope", label = "Slope of the Peak Exercise ST Segment",
                  choices = c("Down", "Flat", "Up"), selected = "Up")
    ),
    # Main panel for displaying predicted results
    mainPanel(
      # Output for predicted results
      h4("Predicted Heart Disease Status:"),
      verbatimTextOutput(outputId = "prediction"),
      plotOutput(outputId = "plot"),
      imageOutput(outputId = "image")
    )
  )
)



# Define server for Shiny app
server <- function(input, output) {
  # Create reactive function for prediction
  predict_heart_disease <- reactive({
    # Create data frame with user input
    new_data <- data.frame(
      Age = input$Age,
      Sex = input$Sex,
      ChestPainType = input$ChestPainType,
      RestingBP = input$RestingBP,
      Cholesterol = input$Cholesterol,
      FastingBS = input$FastingBS,
      RestingECG = input$RestingECG,
      MaxHR = input$MaxHR,
      ExerciseAngina = input$ExerciseAngina,
      Oldpeak = input$Oldpeak,
      ST_Slope = input$ST_Slope
    )
    # Use the trained model to predict heart disease
    pred <- predict(final_rf, new_data)$.pred_class
    # Return the predicted value
    #return(as.character(pred))
    case_when(pred=="Yes"~"Heart Failure Predicted", pred=="No"~"No Event Predicted", T~"Not Predictable")
  })
  
  # Output the predicted heart disease
  output$prediction <- renderText({predict_heart_disease()})
  output$plot <- renderPlot({final_rf %>% extract_fit_parsnip() %>% vip(num_features=30)+labs(title="Variable Importance Weight")})
  output$image <- renderImage({list(src = "roc.png", contentType = 'image/png', width = "100%", height = "100%")}, deleteFile = FALSE)
}

# Run the Shiny app
shinyApp(ui = ui, server = server)





## Testing
# new_data <- data.frame(
#  Age = 50,
#  Sex = "M",
#  ChestPainType = "TA",
#  RestingBP = 130,
#  Cholesterol = 250,
#  FastingBS = "N",
#  RestingECG = "Normal",
#  MaxHR = 150,
#  ExerciseAngina = "N",
#  Oldpeak = 1.5,
#  ST_Slope = "Up"
# )
# predict(final_rf, new_data)$.pred_class


