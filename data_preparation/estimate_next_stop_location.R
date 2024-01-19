# This script estimates a discrete choice model for the module NextStopLocation of the Swiss LCV model.
# The script needs to be adjusted for every branch by changing:
# - apollo_control$modelName
# - apollo_fixed (parameters whose values are kept to the default value (0))
# - mnl_settings$rows (only observations of the desired branch should be considered)

# author: Raphael Ancel (ARE)

# ################################################################# #
#### LOAD LIBRARY AND DEFINE CORE SETTINGS                       ####
# ################################################################# #

### Clear memory
rm(list = ls())

### Load Apollo library
library(apollo)
library(texreg)

### Initialise code
apollo_initialise()

### Set core controls
apollo_control = list(
  modelName       = "next_stop_location_other",
  modelDescr      = " ",
  indivID         = "OID", 
  outputDirectory = "../outputs/next_stop_location/",
  weights         = "STATISTICAL_WEIGHT",
  nCores          = 30
)

# ################################################################# #
#### LOAD DATA AND APPLY ANY TRANSFORMATIONS                     ####
# ################################################################# #

database = read.csv(
  "../data/estimation_data_for_next_stop_location.csv",
  header=TRUE,
  sep=';')


# ################################################################# #
#### DEFINE MODEL PARAMETERS                                     ####
# ################################################################# #

### Vector of parameters, including any that are kept fixed in estimation
apollo_beta=c(b_LowDen  = 0.0,
              b_Res     = 0.0,
              b_Inter   = 0.0,
              b_EmpNode = 0.0,
              b_same_ZIP = 0.0,
              b_cost_first = 0.0,
              b_cost_0 = 0.0,
              b_cost_50 = 0.0,
              b_jobs_pop = 1)

### Vector with names (in quotes) of parameters to be kept fixed at their starting value in apollo_beta, use apollo_beta_fixed = c() if none
apollo_fixed = c('b_EmpNode','b_cost_first', 'b_same_ZIP')

# ################################################################# #
#### GROUP AND VALIDATE INPUTS                                   ####
# ################################################################# #

apollo_inputs = apollo_validateInputs()
apollo_inputs$J = 300 # need to retain J (number of alternatives) for use inside apollo_probabilities

# ################################################################# #
#### DEFINE MODEL AND LIKELIHOOD FUNCTION                        ####
# ################################################################# #

apollo_probabilities=function(apollo_beta, apollo_inputs, functionality="estimate"){
  
  ### Attach inputs and detach after function exit
  apollo_attach(apollo_beta, apollo_inputs)
  on.exit(apollo_detach(apollo_beta, apollo_inputs))
  
  ### Create list of probabilities P
  P = list()
  
  ### List of utilities: these must use the same names as in mnl_settings, order is irrelevant
  V = list()
  IsLargerThan <- function(x, y){
    ifelse(x >y, 1, 0)
  }
  
  
  for(j in 1:apollo_inputs$J){
    V[[paste0("alt_",j)]] = 
      b_LowDen  * get(paste0("LowDensity_", j))  +
      b_Res     * get(paste0("Residential_", j)) +
      b_Inter   * get(paste0("Intermediary_", j))  +
      b_EmpNode * get(paste0("EmploymentNode_", j))  +
      #      b_LowDen  * (get(paste0("LAND_USE_", j)) == "L") +
      #      b_Res     * (get(paste0("LAND_USE_", j)) == "R") +
      #      b_Inter   * (get(paste0("LAND_USE_", j)) == "I") +
      #      b_EmpNode * (get(paste0("LAND_USE_", j)) == "C") +
      b_same_ZIP * get(paste0("isInternal_", j)) +
      (1-get(paste0("isInternal_", j)))*((b_cost_first  * IS_FIRST_LEG  + b_cost_0)  * get(paste0("COST_", j))/100 +
      b_cost_50 * IsLargerThan(get(paste0("COST_", j)), 50) * (get(paste0("COST_", j)) - 50)/100)  +
      log(b_jobs_pop * (1 + get(paste0("JOBS_", j))) +
                    (1 + get(paste0("POP_", j)))) +
      log(get(paste0("P_", j))) 
  } 
  
  ### Define settings for MNL model component
  mnl_settings = list(
    alternatives  = setNames(1:apollo_inputs$J, names(V)), 
    avail         = 1,
    choiceVar     = CHOICE,
    utilities     = V,
    #rows          = (BRANCH ==3)) # branch C
    #rows          = (BRANCH ==6)) # branch F
    #rows          = (BRANCH ==7)) # branch G
    #rows          = (BRANCH ==8)) # branch H
    #rows          = (BRANCH ==14)) # branch N
    #rows          = (BRANCH ==100)) # Private
    rows          = ((BRANCH !=3) & (BRANCH !=6) & (BRANCH !=7) & (BRANCH != 8)  & (BRANCH != 14)  & (BRANCH !=100))) # other
  
  ### Compute probabilities using MNL model
  P[["model"]] = apollo_mnl(mnl_settings, functionality)
  
  ### Take product across observation for same individual
  P = apollo_panelProd(P, apollo_inputs, functionality)
  
  P = apollo_weighting(P, apollo_inputs, functionality)
  
  ### Prepare and return outputs of function
  P = apollo_prepareProb(P, apollo_inputs, functionality)
  return(P)
}

# ################################################################# #
#### MODEL ESTIMATION                                            ####
# ################################################################# #

estimate_settings = list()
estimate_settings[["constraints"]]=c("b_jobs_pop > 0")

model = apollo_estimate(apollo_beta, apollo_fixed, apollo_probabilities, apollo_inputs, estimate_settings)

# ################################################################# #
#### MODEL OUTPUTS                                               ####
# ################################################################# #

# ----------------------------------------------------------------- #
#---- FORMATTED OUTPUT (TO SCREEN)                               ----
# ----------------------------------------------------------------- #

apollo_modelOutput(model)

# ----------------------------------------------------------------- #
#---- FORMATTED OUTPUT (TO FILE, using model name)               ----
# ----------------------------------------------------------------- #

apollo_saveOutput(model)

# generate latex output

# texreg output

quicktexregapollo <- function(model =model, wtpest=NULL) {
  
  modelOutput_settings = list(printPVal=T) 
  
  if (is.null(wtpest)) {  estimated <- janitor::clean_names(as.data.frame(apollo_modelOutput(model, modelOutput_settings)))
  } else{
    estimated <- wtpest
    colnames(estimated)<- c("estimate", "rob_s_e", "robt", "p_1_sided_2")
    
  }
  
  
  coefnames <- gsub(pattern = "_[a-z]$", "" ,rownames(estimated))
  
  texout <- createTexreg(coef.names = coefnames , coef = estimated[["estimate"]] , se = estimated[["rob_s_e"]] , pvalues = estimated$p_1_sided_2,
                         gof.names = c("No Observations" , "Log Likelihood (Null)" , "Log Likelihood (Converged)", "Rho-squared") ,
                         gof = c(model[["nObsTot"]] , model[["LL0"]][[1]] , model[["LLout"]][[1]],  model[["rho2_0"]][[1]]) ,
                         gof.decimal = c(FALSE,FALSE,FALSE,TRUE)
  )
  
  
  return(texout)
  
}

model_texreg <- quicktexregapollo(model)

# save output
save(model_texreg, file = paste0(apollo_control$outputDirectory, apollo_control$modelName, "_texmod.Rdata"))



