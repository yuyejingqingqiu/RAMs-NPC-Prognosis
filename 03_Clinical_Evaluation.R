# ==============================================================================
# Script: 03_Clinical_Evaluation.R
# Purpose: Representative clinical evaluation for RAMs prognostic model
# Dataset: synthetic demonstration dataset only
# ==============================================================================

library(survival)
library(rms)
library(dcurves)
library(survIDINRI)
library(timeROC)
library(pROC)
library(ResourceSelection)

cat("[INFO] Loading synthetic demonstration dataset...\n")
df <- read.csv("dummy_cohort.csv", stringsAsFactors = FALSE)

required_cols <- c(
  "PFS_time", "PFS_status", "T_stage", "N_stage", "LDH", "NLR", "GTVnx",
  "RAMs_score", "Rad_score", "Path_score"
)
missing_cols <- setdiff(required_cols, colnames(df))
if (length(missing_cols) > 0) {
  stop(paste("Missing required columns:", paste(missing_cols, collapse = ", ")))
}

df$T_stage <- factor(df$T_stage, levels = c("T1", "T2", "T3", "T4"), ordered = TRUE)
df$N_stage <- factor(df$N_stage, levels = c("N0", "N1", "N2", "N3"), ordered = TRUE)
df$PFS_status <- as.integer(df$PFS_status)

dd <- datadist(df)
options(datadist = "dd")

fit_clinical <- cph(
  Surv(PFS_time, PFS_status) ~ T_stage + N_stage + LDH + NLR + GTVnx,
  data = df,
  x = TRUE,
  y = TRUE,
  surv = TRUE
)

fit_RAMs <- cph(
  Surv(PFS_time, PFS_status) ~ RAMs_score,
  data = df,
  x = TRUE,
  y = TRUE,
  surv = TRUE
)

fit_nomo <- cph(
  Surv(PFS_time, PFS_status) ~ RAMs_score + T_stage + N_stage + LDH + NLR + GTVnx,
  data = df,
  x = TRUE,
  y = TRUE,
  surv = TRUE
)

c_index_clin <- fit_clinical$stats["Dxy"] / 2 + 0.5
c_index_rams <- fit_RAMs$stats["Dxy"] / 2 + 0.5
cat(sprintf("Clinical model C-index: %.3f\n", c_index_clin))
cat(sprintf("RAMs model C-index: %.3f\n", c_index_rams))

# Time-dependent ROC at 36 months.
roc_rams_36 <- timeROC(
  T = df$PFS_time,
  delta = df$PFS_status,
  marker = df$RAMs_score,
  cause = 1,
  weighting = "marginal",
  times = 36,
  iid = TRUE
)
cat(sprintf("3-year time-dependent AUC for RAMs: %.3f\n", roc_rams_36$AUC[2]))

# Binary 36-month endpoint for DeLong and Hosmer-Lemeshow demonstration.
df$event_36 <- ifelse(
  df$PFS_status == 1 & df$PFS_time <= 36,
  1,
  ifelse(df$PFS_time >= 36, 0, NA)
)
df_delong <- df[!is.na(df$event_36), ]

if (nrow(df_delong) >= 5 && length(unique(df_delong$event_36)) == 2) {
  roc_rams <- roc(df_delong$event_36, df_delong$RAMs_score, quiet = TRUE)
  roc_rad <- roc(df_delong$event_36, df_delong$Rad_score, quiet = TRUE)
  roc_path <- roc(df_delong$event_36, df_delong$Path_score, quiet = TRUE)

  cat("[INFO] DeLong test: RAMs vs radiomics\n")
  print(roc.test(roc_rams, roc_rad, method = "delong"))

  cat("[INFO] DeLong test: RAMs vs pathomics\n")
  print(roc.test(roc_rams, roc_path, method = "delong"))

  cat("[INFO] Hosmer-Lemeshow test for RAMs score\n")
  print(hoslem.test(df_delong$event_36, df_delong$RAMs_score, g = min(5, nrow(df_delong))))
} else {
  cat("[WARNING] Synthetic data are insufficient for DeLong or Hosmer-Lemeshow demonstration.\n")
}

# NRI/IDI demonstration.
covs_clinical <- model.matrix(~ T_stage + N_stage + LDH + NLR + GTVnx, data = df)[, -1, drop = FALSE]
covs_RAMs <- model.matrix(~ T_stage + N_stage + LDH + NLR + GTVnx + RAMs_score, data = df)[, -1, drop = FALSE]

cat("[INFO] NRI/IDI demonstration\n")
idi_nri_results <- tryCatch(
  IDI.INF(
    indata = as.matrix(df[, c("PFS_time", "PFS_status")]),
    covs0 = covs_clinical,
    covs1 = covs_RAMs,
    t0 = 36,
    npert = 300
  ),
  error = function(e) {
    message("NRI/IDI demonstration skipped: ", e$message)
    NULL
  }
)
if (!is.null(idi_nri_results)) print(idi_nri_results)

# Calibration curve.
cal_RAMs <- calibrate(fit_RAMs, cmethod = "KM", method = "boot", u = 36, m = 5, B = 100)
pdf("Figure_Calibration_RAMs.pdf", width = 6, height = 6)
plot(
  cal_RAMs,
  lwd = 2,
  xlab = "Predicted 3-year PFS probability",
  ylab = "Observed 3-year PFS probability"
)
dev.off()

# DCA.
df$pred_clinical <- 1 - survest(fit_clinical, newdata = df, times = 36)$surv
df$pred_RAMs <- 1 - survest(fit_RAMs, newdata = df, times = 36)$surv

dca_results <- dca(
  Surv(PFS_time, PFS_status) ~ pred_clinical + pred_RAMs,
  data = df,
  time = 36,
  thresholds = seq(0.01, 0.85, by = 0.01)
)
pdf("Figure_DCA_Comparison.pdf", width = 7, height = 6)
plot(dca_results)
dev.off()

# Nomogram.
surv_fun <- Survival(fit_nomo)
surv_1yr <- function(x) surv_fun(12, x)
surv_3yr <- function(x) surv_fun(36, x)

nomogram_obj <- nomogram(
  fit_nomo,
  fun = list(surv_1yr, surv_3yr),
  lp = TRUE,
  funlabel = c("1-year PFS probability", "3-year PFS probability")
)

pdf("Figure_Nomogram_RAMs.pdf", width = 10, height = 7)
plot(nomogram_obj, cex.axis = 0.8, xfrac = 0.4)
dev.off()

cat("[SUCCESS] Clinical evaluation demonstration completed.\n")
