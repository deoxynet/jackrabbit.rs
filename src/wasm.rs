/// ggwave-compatible WASM bindings.
///
/// API contract:
/// ```javascript
/// ggwave_factory() → Promise<module>
/// module.getDefaultParameters() → { sampleRateInp, sampleRateOut, sampleRate, ... }
/// module.init(params) → handle (integer)
/// module.encode(handle, Uint8Array, protocolId, volume) → Int8Array
/// module.decode(handle, Int8Array) → Uint8Array
/// module.free(handle)
/// module.ProtocolId  // 12 enum values
/// module.SampleFormat // GGWAVE_SAMPLE_FORMAT_F32
/// ```

use wasm_bindgen::prelude::*;
use js_sys::{self, Object, Reflect, Uint8Array, Int8Array};
use std::cell::RefCell;
use std::collections::HashMap;

use crate::decoder::Decoder;
use crate::encoder;
use crate::params::{SAMPLE_RATE, PROTOCOLS};

struct Instance {
    decoder: Decoder,
}

thread_local! {
    static INSTANCES: RefCell<HashMap<u32, Instance>> = RefCell::new(HashMap::new());
    static NEXT_ID: RefCell<u32> = RefCell::new(1);
}

/// Create the module object that mimics ggwave's API.
#[wasm_bindgen(js_name = "ggwave_factory")]
pub async fn ggwave_factory() -> Result<JsValue, JsValue> {
    let module = Object::new();

    // getDefaultParameters
    let get_default_params = Closure::wrap(Box::new(|| -> JsValue {
        let params = Object::new();
        Reflect::set(&params, &"sampleRateInp".into(), &JsValue::from(SAMPLE_RATE)).unwrap();
        Reflect::set(&params, &"sampleRateOut".into(), &JsValue::from(SAMPLE_RATE)).unwrap();
        Reflect::set(&params, &"sampleRate".into(), &JsValue::from(SAMPLE_RATE)).unwrap();
        Reflect::set(&params, &"samplesPerFrame".into(), &JsValue::from(1024)).unwrap();
        Reflect::set(&params, &"sampleFormatInp".into(), &JsValue::from(1)).unwrap(); // F32
        Reflect::set(&params, &"sampleFormatOut".into(), &JsValue::from(1)).unwrap(); // F32
        params.into()
    }) as Box<dyn Fn() -> JsValue>);
    Reflect::set(&module, &"getDefaultParameters".into(), get_default_params.as_ref())?;
    get_default_params.forget();

    // init(params) → handle
    let init_fn = Closure::wrap(Box::new(|_params: JsValue| -> u32 {
        NEXT_ID.with(|id| {
            let handle = *id.borrow();
            *id.borrow_mut() = handle + 1;
            INSTANCES.with(|instances| {
                instances.borrow_mut().insert(handle, Instance {
                    decoder: Decoder::new(),
                });
            });
            handle
        })
    }) as Box<dyn Fn(JsValue) -> u32>);
    Reflect::set(&module, &"init".into(), init_fn.as_ref())?;
    init_fn.forget();

    // encode(handle, Uint8Array, protocolId, volume) → Int8Array
    let encode_fn = Closure::wrap(Box::new(|_handle: u32, data: Uint8Array, protocol_id: u32, volume: f32| -> Int8Array {
        let payload = data.to_vec();
        if payload.is_empty() || payload.len() > 1023 || protocol_id >= PROTOCOLS.len() as u32 {
            return Int8Array::new_with_length(0);
        }

        let waveform = encoder::encode(&payload, protocol_id as u8, volume);

        // Convert f32 samples to byte view (Int8Array over the same f32 data)
        let byte_len = waveform.len() * 4;

        // Safety: reinterpret f32 slice as u8 slice
        let bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                waveform.as_ptr() as *const u8,
                byte_len,
            )
        };
        // Int8Array::copy_from expects &[i8], but we have &[u8].
        // Use the underlying buffer directly.
        let u8_array = Uint8Array::new_with_length(byte_len as u32);
        u8_array.copy_from(bytes);
        Int8Array::new(&u8_array.buffer())
    }) as Box<dyn Fn(u32, Uint8Array, u32, f32) -> Int8Array>);
    Reflect::set(&module, &"encode".into(), encode_fn.as_ref())?;
    encode_fn.forget();

    // decode(handle, Int8Array) → Uint8Array
    let decode_fn = Closure::wrap(Box::new(|handle: u32, data: Int8Array| -> Uint8Array {
        // Reinterpret Int8Array as f32 samples
        let byte_len = data.length() as usize;
        if byte_len == 0 || byte_len % 4 != 0 {
            return Uint8Array::new_with_length(0);
        }

        let u8_view = Uint8Array::new_with_byte_offset_and_length(
            &data.buffer(),
            data.byte_offset(),
            data.length(),
        );
        let bytes = u8_view.to_vec();

        let samples: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|chunk| {
                let arr: [u8; 4] = [chunk[0], chunk[1], chunk[2], chunk[3]];
                f32::from_le_bytes(arr)
            })
            .collect();

        let result = INSTANCES.with(|instances| {
            let mut map = instances.borrow_mut();
            if let Some(instance) = map.get_mut(&handle) {
                instance.decoder.feed(&samples)
            } else {
                Vec::new()
            }
        });

        if result.is_empty() {
            Uint8Array::new_with_length(0)
        } else {
            let out = Uint8Array::new_with_length(result.len() as u32);
            out.copy_from(&result);
            out
        }
    }) as Box<dyn Fn(u32, Int8Array) -> Uint8Array>);
    Reflect::set(&module, &"decode".into(), decode_fn.as_ref())?;
    decode_fn.forget();

    // debugInfo(handle) → { state, bufferLen, totalSamples, maxAmplitude, rmsAmplitude }
    let debug_fn = Closure::wrap(Box::new(|handle: u32| -> JsValue {
        INSTANCES.with(|instances| {
            let map = instances.borrow();
            if let Some(instance) = map.get(&handle) {
                let info = instance.decoder.debug_info();
                let obj = Object::new();
                Reflect::set(&obj, &"state".into(), &JsValue::from(info.state)).unwrap();
                Reflect::set(&obj, &"bufferLen".into(), &JsValue::from(info.buffer_len as u32)).unwrap();
                Reflect::set(&obj, &"totalSamples".into(), &JsValue::from(info.total_samples_fed as f64)).unwrap();
                Reflect::set(&obj, &"maxAmplitude".into(), &JsValue::from(info.max_amplitude as f64)).unwrap();
                Reflect::set(&obj, &"rmsAmplitude".into(), &JsValue::from(info.rms_amplitude as f64)).unwrap();
                Reflect::set(&obj, &"lastEvent".into(), &JsValue::from(info.last_event)).unwrap();
                Reflect::set(&obj, &"preambleCount".into(), &JsValue::from(info.preamble_count)).unwrap();
                obj.into()
            } else {
                JsValue::NULL
            }
        })
    }) as Box<dyn Fn(u32) -> JsValue>);
    Reflect::set(&module, &"debugInfo".into(), debug_fn.as_ref())?;
    debug_fn.forget();

    // free(handle)
    let free_fn = Closure::wrap(Box::new(|handle: u32| {
        INSTANCES.with(|instances| {
            instances.borrow_mut().remove(&handle);
        });
    }) as Box<dyn Fn(u32)>);
    Reflect::set(&module, &"free".into(), free_fn.as_ref())?;
    free_fn.forget();

    // ProtocolId enum
    let protocol_id = Object::new();
    Reflect::set(&protocol_id, &"GGWAVE_PROTOCOL_AUDIBLE_NORMAL".into(), &JsValue::from(0))?;
    Reflect::set(&protocol_id, &"GGWAVE_PROTOCOL_AUDIBLE_FAST".into(), &JsValue::from(1))?;
    Reflect::set(&protocol_id, &"GGWAVE_PROTOCOL_AUDIBLE_FASTEST".into(), &JsValue::from(2))?;
    Reflect::set(&protocol_id, &"GGWAVE_PROTOCOL_DT_NORMAL".into(), &JsValue::from(3))?;
    Reflect::set(&protocol_id, &"GGWAVE_PROTOCOL_DT_FAST".into(), &JsValue::from(4))?;
    Reflect::set(&protocol_id, &"GGWAVE_PROTOCOL_DT_FASTEST".into(), &JsValue::from(5))?;
    Reflect::set(&protocol_id, &"GGWAVE_PROTOCOL_MT_NORMAL".into(), &JsValue::from(6))?;
    Reflect::set(&protocol_id, &"GGWAVE_PROTOCOL_MT_FAST".into(), &JsValue::from(7))?;
    Reflect::set(&protocol_id, &"GGWAVE_PROTOCOL_WIRED_FAST".into(), &JsValue::from(8))?;
    Reflect::set(&protocol_id, &"GGWAVE_PROTOCOL_UT_NORMAL".into(), &JsValue::from(9))?;
    Reflect::set(&protocol_id, &"GGWAVE_PROTOCOL_UT_FAST".into(), &JsValue::from(10))?;
    Reflect::set(&protocol_id, &"GGWAVE_PROTOCOL_UT_FASTEST".into(), &JsValue::from(11))?;
    Reflect::set(&protocol_id, &"GGWAVE_PROTOCOL_WIRED_TURBO".into(), &JsValue::from(13))?;
    Reflect::set(&protocol_id, &"GGWAVE_PROTOCOL_WIRED_MAX".into(), &JsValue::from(14))?;
    Reflect::set(&protocol_id, &"GGWAVE_PROTOCOL_OTA_ROBUST".into(), &JsValue::from(15))?;
    Reflect::set(&module, &"ProtocolId".into(), &protocol_id)?;

    // SampleFormat
    let sample_format = Object::new();
    Reflect::set(&sample_format, &"GGWAVE_SAMPLE_FORMAT_F32".into(), &JsValue::from(1))?;
    Reflect::set(&module, &"SampleFormat".into(), &sample_format)?;

    Ok(module.into())
}

#[cfg(all(test, feature = "wasm", target_arch = "wasm32"))]
mod tests {
    use super::ggwave_factory;
    use js_sys::{Function, Int8Array, Reflect, Uint8Array};
    use wasm_bindgen::JsCast;
    use wasm_bindgen::JsValue;
    use wasm_bindgen_test::wasm_bindgen_test;

    #[wasm_bindgen_test(async)]
    async fn test_decode_respects_subarray_offset() {
        let module = ggwave_factory().await.expect("factory should build module");

        let init = Reflect::get(&module, &JsValue::from_str("init"))
            .unwrap()
            .dyn_into::<Function>()
            .unwrap();
        let encode = Reflect::get(&module, &JsValue::from_str("encode"))
            .unwrap()
            .dyn_into::<Function>()
            .unwrap();
        let decode = Reflect::get(&module, &JsValue::from_str("decode"))
            .unwrap()
            .dyn_into::<Function>()
            .unwrap();
        let free = Reflect::get(&module, &JsValue::from_str("free"))
            .unwrap()
            .dyn_into::<Function>()
            .unwrap();

        let handle = init
            .call1(&module, &JsValue::NULL)
            .unwrap()
            .as_f64()
            .unwrap() as u32;

        let payload_bytes = b"subarray";
        let payload = Uint8Array::from(payload_bytes.as_slice());
        let encoded = encode
            .call4(
                &module,
                &JsValue::from(handle),
                &payload,
                &JsValue::from(0u32),
                &JsValue::from_f64(1.0),
            )
            .unwrap()
            .dyn_into::<Int8Array>()
            .unwrap();

        let encoded_len = encoded.length() as usize;
        assert!(encoded_len > 0);

        let encoded_bytes: Vec<u8> = encoded.to_vec().into_iter().map(|b| b as u8).collect();
        let prefix = 7usize;
        let suffix = 5usize;
        let mut backing = vec![0u8; prefix + encoded_len + suffix];
        backing[prefix..prefix + encoded_len].copy_from_slice(&encoded_bytes);

        let backing_u8 = Uint8Array::from(backing.as_slice());
        let sub = Int8Array::new_with_byte_offset_and_length(
            &backing_u8.buffer(),
            prefix as u32,
            encoded_len as u32,
        );

        let decoded = decode
            .call2(&module, &JsValue::from(handle), &JsValue::from(sub))
            .unwrap()
            .dyn_into::<Uint8Array>()
            .unwrap();

        assert_eq!(decoded.to_vec(), payload_bytes.to_vec());

        free.call1(&module, &JsValue::from(handle)).unwrap();
    }
}
