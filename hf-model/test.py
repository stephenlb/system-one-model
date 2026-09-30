from transformers import pipeline
pipe = pipeline('system-one', model='stephenlb/system-one-model',
trust_remote_code=True, dtype='bfloat16', device_map='auto')

print(pipe({
    'state': 'I was charged twice.',
    'questions': {
        'refund': {
            'type': 'noul',
            'instructions': 'Does the text request a refund?'
        }
    }
}))
