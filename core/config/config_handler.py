importos
importuuid

importtoml
importyaml
fromfiber.logging_utilsimportget_logger
fromtransformersimportAutoTokenizer

importcore.constantsascst
fromcore.models.utility_modelsimportTextDatasetType
fromcore.models.utility_modelsimportDpoDatasetType
fromcore.models.utility_modelsimportFileFormat
fromcore.models.utility_modelsimportGrpoDatasetType
fromcore.models.utility_modelsimportInstructTextDatasetType
fromcore.models.utility_modelsimportChatTemplateDatasetType


logger=get_logger(__name__)


defcreate_dataset_entry(
dataset:str,
dataset_type:TextDatasetType,
file_format:FileFormat,
is_eval:bool=False,
)->dict:
dataset_entry={"path":dataset}

logger.info(dataset_type)

iffile_format==FileFormat.JSON:
ifnotis_eval:
dataset_entry={"path":"/workspace/input_data/"}
else:
dataset_entry={"path":f"/workspace/input_data/{os.path.basename(dataset)}"}

ifisinstance(dataset_type,InstructTextDatasetType):
instruct_type_dict={key:valueforkey,valueindataset_type.model_dump().items()ifvalueisnotNone}
dataset_entry.update(_process_instruct_dataset_fields(instruct_type_dict))
elifisinstance(dataset_type,DpoDatasetType):
dataset_entry.update(_process_dpo_dataset_fields(dataset_type))
elifisinstance(dataset_type,GrpoDatasetType):
dataset_entry.update(_process_grpo_dataset_fields(dataset_type))
elifisinstance(dataset_type,ChatTemplateDatasetType):
dataset_entry.update(_process_chat_template_dataset_fields(dataset_type))
else:
raiseValueError("Invaliddataset_typeprovided.")

iffile_format!=FileFormat.HF:
dataset_entry["ds_type"]=file_format.value
dataset_entry["data_files"]=[os.path.basename(dataset)]

returndataset_entry


defupdate_flash_attention(config:dict,model:str):
#Youmightwanttomakethismodel-dependent
config["flash_attention"]=False
returnconfig


defupdate_model_info(config:dict,model:str,job_id:str="",expected_repo_name:str|None=None):
logger.info("WEAREUPDATINGTHEMODELINFO")
tokenizer=AutoTokenizer.from_pretrained(model,trust_remote_code=True)
iftokenizer.pad_token_idisNoneandtokenizer.eos_token_idisnotNone:
config["special_tokens"]={"pad_token":tokenizer.eos_token}

config["base_model"]=model
config["wandb_runid"]=job_id
config["wandb_name"]=job_id
config["hub_model_id"]=f"{cst.HUGGINGFACE_USERNAME}/{expected_repo_nameorstr(uuid.uuid4())}"

returnconfig


defsave_config(config:dict,config_path:str):
withopen(config_path,"w")asfile:
yaml.dump(config,file)


defsave_config_toml(config:dict,config_path:str):
withopen(config_path,"w")asfile:
toml.dump(config,file)


def_process_grpo_dataset_fields(dataset_type:GrpoDatasetType)->dict:
return{"split":"train"}


def_process_dpo_dataset_fields(dataset_type:DpoDatasetType)->dict:
#Enablebelowwhenhttps://github.com/axolotl-ai-cloud/axolotl/issues/1417isfixed
#context:https://discord.com/channels/1272221995400167588/1355226588178022452/1356982842374226125

#dpo_type_dict=dataset_type.model_dump()
#dpo_type_dict["type"]="user_defined.default"
#ifnotdpo_type_dict.get("prompt_format"):
#ifdpo_type_dict.get("field_system"):
#dpo_type_dict["prompt_format"]="{system}{prompt}"
#else:
#dpo_type_dict["prompt_format"]="{prompt}"
#returndpo_type_dict

#Fallbacktohttps://axolotl-ai-cloud.github.io/axolotl/docs/rlhf.html#chatml.intel
#Columnnamesarehardcodedinaxolotl:"DPO_DEFAULT_FIELD_SYSTEM",
#"DPO_DEFAULT_FIELD_PROMPT","DPO_DEFAULT_FIELD_CHOSEN","DPO_DEFAULT_FIELD_REJECTED"
return{"type":cst.DPO_DEFAULT_DATASET_TYPE,"split":"train"}


def_process_instruct_dataset_fields(instruct_type_dict:dict)->dict:
ifnotinstruct_type_dict.get("field_output"):
return{
"type":"completion",
"field":instruct_type_dict.get("field_instruction"),
}

processed_dict=instruct_type_dict.copy()
processed_dict.setdefault("no_input_format","{instruction}")
ifprocessed_dict.get("field_input"):
processed_dict.setdefault("format","{instruction}{input}")
else:
processed_dict.setdefault("format","{instruction}")

return{"format":"custom","type":processed_dict}


def_process_chat_template_dataset_fields(dataset_dict:dict)->dict:
processed_dict={}

processed_dict["chat_template"]=dataset_dict.chat_template
processed_dict["type"]="chat_template"
processed_dict["field_messages"]=dataset_dict.chat_column
processed_dict["message_field_role"]=dataset_dict.chat_role_field
processed_dict["message_field_content"]=dataset_dict.chat_content_field
processed_dict["roles"]={
"assistant":[dataset_dict.chat_assistant_reference],
"user":[dataset_dict.chat_user_reference],
}

processed_dict["message_property_mappings"]={
"role":dataset_dict.chat_role_field,
"content":dataset_dict.chat_content_field
}

returnprocessed_dict
