# ==============================================================================
# Script: 03_Clinical_Evaluation.R
# Purpose: Clinical evaluation, model comparison, and Nomogram construction
# Metrics: C-index, Time-dependent ROC, Calibration, DCA, NRI, IDI
# Language: R (version 4.2.1+)
# ==============================================================================

# Load necessary medical statistics packages
# (These are the gold standard packages for high-impact oncology papers)
library(survival)      # Core survival analysis
library(rms)           # Regression Modeling Strategies (for Nomogram and Calibration)
library(dcurves)       # Decision Curve Analysis (DCA)
library(survIDINRI)    # Continuous NRI and IDI for survival data
library(timeROC)       # Time-dependent ROC curves

# ------------------------------------------------------------------------------
# 1. Data Loading & Preprocessing
# ------------------------------------------------------------------------------
# Note: In the public repository, we do not upload the real patient dataset 
# due to privacy regulations. This script assumes the existence of a clean CSV.
# 
# The dataset is expected to contain:
# - PFS_time: Progression-Free Survival time (in months)
# - PFS_status: Event indicator (1 = Progression/Death, 0 = Censored)
# - Clinical predictors: T_stage, N_stage, LDH, NLR, GTVnx
# - Model outputs: RAMs_score, Rad_score, Path_score
# ------------------------------------------------------------------------------

cat("[INFO] Loading cohort dataset...\n")
# Example path: df <- read.csv("path/to/your/real_results_data.csv")
# Here we define the pipeline using 'df' as the dataframe object.
df <- read.csv("Cohort_Results_Data.csv") 

# Define global distribution settings for rms package (Crucial for Nomogram)
dd <- datadist(df)
options(datadist = "dd")

# ------------------------------------------------------------------------------
# 2. Model Construction (Cox Proportional Hazards)
# ------------------------------------------------------------------------------
cat("[INFO] Fitting Cox proportional hazards models...\n")

# Baseline Clinical Model
fit_clinical <- cph(Surv(PFS_time, PFS_status) ~ T_stage + N_stage + LDH + NLR + GTVnx, 
                    data = df, x = TRUE, y = TRUE, surv = TRUE)

# Multimodal RAMs Fusion Model
fit_RAMs <- cph(Surv(PFS_time, PFS_status) ~ RAMs_score, 
                data = df, x = TRUE, y = TRUE, surv = TRUE)

# ------------------------------------------------------------------------------
# 3. Model Performance Evaluation: C-index & Time-dependent AUC
# ------------------------------------------------------------------------------
cat("[INFO] Calculating C-index...\n")

c_index_clin <- fit_clinical$stats["Dxy"] / 2 + 0.5
c_index_rams <- fit_RAMs$stats["Dxy"] / 2 + 0.5

cat(sprintf("Clinical Model C-index: %.3f\n", c_index_clin))
cat(sprintf("RAMs Fusion Model C-index: %.3f\n", c_index_rams))

# Calculate 3-year Time-dependent ROC
# Assuming time is in months, 3 years = 36 months
ROC_3yr <- timeROC(T = df$PFS_time, 
                   delta = df$PFS_status, 
                   marker = df$RAMs_score, 
                   cause = 1, weighting = "marginal", 
                   times = 36, iid = TRUE)
cat(sprintf("3-Year AUC for RAMs Model: %.3f\n", ROC_3yr$AUC[2]))

# ------------------------------------------------------------------------------
# 4. Incremental Value: Continuous NRI & IDI
# ------------------------------------------------------------------------------
cat("[INFO] Calculating Continuous NRI and IDI for 3-Year PFS...\n")

# Compare Clinical Baseline Model vs. RAMs Fusion Model
# The survIDINRI package requires matrix inputs for covariates
covs_clinical <- as.matrix(df[, c("T_stage", "N_stage", "LDH", "NLR", "GTVnx")])
covs_RAMs <- as.matrix(df[, c("T_stage", "N_stage", "LDH", "NLR", "GTVnx", "RAMs_score")])

idi_nri_results <- IDI.INF(indata = df[, c("PFS_time", "PFS_status")], 
                           covs0 = covs_clinical, 
                           covs1 = covs_RAMs, 
                           t0 = 36, 
                           npert = 300) # 300 perturbations for robust CI

cat("Incremental Value (RAMs vs Clinical):\n")
print(idi_nri_results)

# ------------------------------------------------------------------------------
# 5. Goodness-of-fit: Calibration Curves
# ------------------------------------------------------------------------------
cat("[INFO] Generating 3-Year Calibration Curves...\n")

# Calculate calibration for the RAMs model using bootstrapping
cal_RAMs <- calibrate(fit_RAMs, cmethod = "KM", method = "boot", 
                      u = 36, m = 50, B = 1000)

# Plotting the calibration curve
pdf("Figure_Calibration_RAMs.pdf", width = 6, height = 6)
plot(cal_RAMs, lwd = 2, lty = 1,
     xlab = "Nomogram Predicted 3-Year PFS Probability",
     ylab = "Actual 3-Year PFS Proportion",
     main = "Calibration Curve of RAMs Model")
dev.off()

# ------------------------------------------------------------------------------
# 6. Clinical Utility: Decision Curve Analysis (DCA)
# ------------------------------------------------------------------------------
cat("[INFO] Performing Decision Curve Analysis (DCA)...\n")

# Compute 3-year survival probabilities for DCA
df$pred_clinical <- 1 - survest(fit_clinical, newdata = df, times = 36)$surv
df$pred_RAMs <- 1 - survest(fit_RAMs, newdata = df, times = 36)$surv

# Run DCA using the dcurves package
dca_results <- dca(Surv(PFS_time, PFS_status) ~ pred_clinical + pred_RAMs, 
                   data = df, 
                   time = 36, 
                   thresholds = seq(0.01, 0.85, by = 0.01))

# Plotting DCA
pdf("Figure_DCA_Comparison.pdf", width = 7, height = 6)
plot(dca_results, 
     ylab = "Net Benefit", 
     xlab = "Threshold Probability",
     main = "Decision Curve Analysis (3-Year PFS)")
dev.off()

# ------------------------------------------------------------------------------
# 7. Bedside Clinical Application: Visual Nomogram
# ------------------------------------------------------------------------------
cat("[INFO] Constructing Individualized Nomogram...\n")

# Create a comprehensive model combining clinical factors and RAMs score
fit_nomo <- cph(Surv(PFS_time, PFS_status) ~ RAMs_score + T_stage + N_stage + LDH + NLR, 
                data = df, x = TRUE, y = TRUE, surv = TRUE)

# Generate the survival probability mapping functions
surv_1yr <- function(x) surv(12, x)
surv_3yr <- function(x) surv(36, x)

# Build the Nomogram object
nomogram_obj <- nomogram(fit_nomo, 
                         fun = list(surv_1yr, surv_3yr), 
                         lp = TRUE, 
                         funlabel = c("1-Year PFS Prob.", "3-Year PFS Prob."), 
                         maxscale = 100, 
                         fun.at = c(0.99, 0.95, 0.90, 0.80, 0.60, 0.40, 0.20, 0.10))

# Plotting the Nomogram
pdf("Figure_Nomogram_RAMs.pdf", width = 10, height = 7)
plot(nomogram_obj, cex.axis = 0.8, xfrac = 0.4)
dev.off()

cat("[SUCCESS] Clinical evaluation pipeline completed. All figures saved.\n")
# ==============================================================================
# End of Script
# ==============================================================================