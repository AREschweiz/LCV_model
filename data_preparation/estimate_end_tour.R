# This script estimates a discrete choice model for the module EndTour of the Swiss LCV model.
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

### Initialise code
apollo_initialise()

### Set core controls
apollo_control = list(
  modelName       = "EndTour_Other",
  modelDescr      = "Binomial logit model on LWE EndTour",
  indivID         = "OID", 
  outputDirectory = "../outputs/end_tour/",
  weights         = "STATISTICAL_WEIGHT"
)

# ################################################################# #w
#### LOAD DATA AND APPLY ANY TRANSFORMATIONS                     ####
# ################################################################# #

### Loading data from package
### if data is to be loaded from a file (e.g. called data.csv), 
### the code would be: database = read.csv("data.csv",header=TRUE)
database = read.csv("../data/estimation_data_for_end_tour.csv", header=TRUE, sep=';')

# ################################################################# #
#### DEFINE MODEL PARAMETERS                                     ####
# ################################################################# #

### Vector of parameters, including any that are kept fixed in estimation
apollo_beta=c(ASC=0,
              cons_2stops = 0,
              b_lnStops = 0,
              b_cost_return = 0)

### Vector with names (in quotes) of parameters to be kept fixed at their starting value in apollo_beta, use apollo_beta_fixed = c() if none
apollo_fixed = c('b_lnStops', 'b_cost_return')
#apollo_fixed = c('b_cost_return')
#apollo_fixed = c('b_lnStops')
#apollo_fixed = c('cons_2stops', 'b_cost_return')
#apollo_fixed = c()

# ################################################################# #
#### GROUP AND VALIDATE INPUTS                                   ####
# ################################################################# #

apollo_inputs = apollo_validateInputs()

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
  V[["return"]]	  = 0
  V[["continue"]] = ASC + 
    cons_2stops * (TRIP_ID == 1) +
    b_lnStops * log(TRIP_ID+1) +
    b_cost_return * COST_RETURN/100
  
    mnl_settings = list(
    alternatives  = c(return=1, continue=0), 
    avail         = 1,
    choiceVar     = IS_RETURN,
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
estimate_settings[["constraints"]]=c()

model = apollo_estimate(apollo_beta, apollo_fixed, apollo_probabilities, apollo_inputs, estimate_settings)

# ################################################################# #
#### MODEL OUTPUTS                                               ####
# ################################################################# #

# ----------------------------------------------------------------- #
#---- FORMATTED OUTPUT (TO SCREEN)                               ----
# ----------------------------------------------------------------- #
modelOutput_settings = list()
modelOutput_settings['printPVal'] =1

apollo_modelOutput(model, modelOutput_settings)

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



