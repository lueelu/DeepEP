"""CPU-only parser for the independent LL dispatch reference-cycle format.

Timestamp subtraction is performed on Python integers before unit conversion.
Presence comes from the validity mask: an executed zero is distinct from absence.
"""
import math

FORMAT = 'll-reference-cycles-v1'
SCHEMA = 4
MAGIC = 0x4C4C4350
SLOTS, FIELDS, WORD_COUNT = 80, 8, 640
DEFAULT_FREQUENCY_HZ = 1_000_000_000
WORDS = ('kernel_entry', 'config_ready', 'magic_begin', 'magic_end', 'counts_init_end', 'role_init_end', 'send_count_begin', 'send_count_issue_end', 'sender_begin', 'sender_end_after_MTE3_S', 'recv_count_begin', 'recv_count_end', 'cumsum_begin', 'cumsum_issue_end', 'complete_cumsum_begin', 'complete_cumsum_issue_end', 'count_drain_begin', 'count_drain_end_PIPE_ALL', 'receiver_begin', 'receiver_end', 'final_drain_begin', 'final_drain_end_PIPE_ALL', 'reserved', 'reserved', 'count_ready_poll_begin', 'count_ready_poll_end_after_V_S', 'cumsum_peer_join_begin', 'cumsum_peer_join_clear_issue_end', 'prefix_ready_wait_begin', 'prefix_ready_wait_clear_issue_end', 'receive_prefix_loaded_after_MTE2_S', 'receive_tasks_built', 'receive_buffers_ready', 'receive_packet_poll_accum_cycles', 'receive_load_consume_submit_accum_cycles', 'receive_lane_wait_clear_issue_end', 'receive_final_PIPE_ALL_end', 'receive_task_count', 'send_buffers_routes_ready', 'send_packets_issue_end', 'send_metadata_end_after_MTE3_S', 'receive_probe_batch_count', 'first_packet_ready', 'send_route_count', 'send_count_first_frame_issued', 'send_count_last_frame_issued', 'recv_count_first_ready', 'recv_count_poll_first_iter_end', 'receive_first_hop_tokens', 'receive_relay_members', 'rank', 'AIV_core', 'EP', 'active_aiv', 'BS', 'schema', 'operation', 'before_finalize', 'valid_mask', 'magic', 'boundary_mode', 'complete', 'receive_relay_distinct_ranks', 'reserved', 'packet_detail_version', 'send_remote_packets', 'send_group_packets', 'remote_first_begin', 'remote_last_put_return', 'group_first_begin', 'group_last_submit_return', 'second_remote_begin', 'second_remote_enqueue', 'second_remote_put_begin', 'second_remote_put_return', 'second_remote_member_q_ticks', 'second_remote_token', 'second_remote_k', 'second_remote_expert', 'second_remote_peer', 'batch_quiet_ticks', 'batch_flush_count', 'last_quiet_end', 'second_remote_triggered_flush', 'remote_first_order', 'remote_last_order', 'group_first_order', 'group_last_order', 'send_packet_count', 'send_lanes', 'second_stage_queued', 'second_stage_wait_begin', 'second_stage_wait_end', 'deferred_remote_packets', 'second_previous_submit_ticks', 'second_post_deferred', 'recv_trace_rows', 'recv_trace_capacity', 'recv_trace_total_tasks', 'recv_trace_ready_seen', 'recv_trace_processed_seen', 'recv_trace_lanes', 'recv_trace_export_begin', 'recv_trace_export_end', 'recv_trace_row_begin', 'recv_trace_bs', 'recv_trace_local_experts', 'recv_trace_truncated', 'detail_reserved', 'detail_reserved', 'detail_reserved', 'detail_reserved', 'detail_reserved', 'detail_reserved', 'detail_reserved', 'detail_reserved', 'detail_reserved', 'detail_reserved', 'detail_reserved', 'detail_reserved', 'detail_reserved', 'detail_reserved', 'detail_reserved', 'detail_reserved', 'detail_reserved', 'detail_reserved', 'detail_reserved', 'detail_reserved', 'packet_log_word_0', 'packet_log_word_1', 'packet_log_word_2', 'packet_log_word_3', 'packet_log_word_4', 'packet_log_word_5', 'packet_log_word_6', 'packet_log_word_7', 'packet_log_word_8', 'packet_log_word_9', 'packet_log_word_10', 'packet_log_word_11', 'packet_log_word_12', 'packet_log_word_13', 'packet_log_word_14', 'packet_log_word_15', 'packet_log_word_16', 'packet_log_word_17', 'packet_log_word_18', 'packet_log_word_19', 'packet_log_word_20', 'packet_log_word_21', 'packet_log_word_22', 'packet_log_word_23', 'packet_log_word_24', 'packet_log_word_25', 'packet_log_word_26', 'packet_log_word_27', 'packet_log_word_28', 'packet_log_word_29', 'packet_log_word_30', 'packet_log_word_31', 'packet_log_word_32', 'packet_log_word_33', 'packet_log_word_34', 'packet_log_word_35', 'packet_log_word_36', 'packet_log_word_37', 'packet_log_word_38', 'packet_log_word_39', 'packet_log_word_40', 'packet_log_word_41', 'packet_log_word_42', 'packet_log_word_43', 'packet_log_word_44', 'packet_log_word_45', 'packet_log_word_46', 'packet_log_word_47', 'packet_log_word_48', 'packet_log_word_49', 'packet_log_word_50', 'packet_log_word_51', 'packet_log_word_52', 'packet_log_word_53', 'packet_log_word_54', 'packet_log_word_55', 'packet_log_word_56', 'packet_log_word_57', 'packet_log_word_58', 'packet_log_word_59', 'packet_log_word_60', 'packet_log_word_61', 'packet_log_word_62', 'packet_log_word_63', 'packet_log_word_64', 'packet_log_word_65', 'packet_log_word_66', 'packet_log_word_67', 'packet_log_word_68', 'packet_log_word_69', 'packet_log_word_70', 'packet_log_word_71', 'packet_log_word_72', 'packet_log_word_73', 'packet_log_word_74', 'packet_log_word_75', 'packet_log_word_76', 'packet_log_word_77', 'packet_log_word_78', 'packet_log_word_79', 'packet_log_word_80', 'packet_log_word_81', 'packet_log_word_82', 'packet_log_word_83', 'packet_log_word_84', 'packet_log_word_85', 'packet_log_word_86', 'packet_log_word_87', 'packet_log_word_88', 'packet_log_word_89', 'packet_log_word_90', 'packet_log_word_91', 'packet_log_word_92', 'packet_log_word_93', 'packet_log_word_94', 'packet_log_word_95', 'packet_log_word_96', 'packet_log_word_97', 'packet_log_word_98', 'packet_log_word_99', 'packet_log_word_100', 'packet_log_word_101', 'packet_log_word_102', 'packet_log_word_103', 'packet_log_word_104', 'packet_log_word_105', 'packet_log_word_106', 'packet_log_word_107', 'packet_log_word_108', 'packet_log_word_109', 'packet_log_word_110', 'packet_log_word_111', 'packet_log_word_112', 'packet_log_word_113', 'packet_log_word_114', 'packet_log_word_115', 'packet_log_word_116', 'packet_log_word_117', 'packet_log_word_118', 'packet_log_word_119', 'packet_log_word_120', 'packet_log_word_121', 'packet_log_word_122', 'packet_log_word_123', 'packet_log_word_124', 'packet_log_word_125', 'packet_log_word_126', 'packet_log_word_127', 'packet_log_word_128', 'packet_log_word_129', 'packet_log_word_130', 'packet_log_word_131', 'packet_log_word_132', 'packet_log_word_133', 'packet_log_word_134', 'packet_log_word_135', 'packet_log_word_136', 'packet_log_word_137', 'packet_log_word_138', 'packet_log_word_139', 'packet_log_word_140', 'packet_log_word_141', 'packet_log_word_142', 'packet_log_word_143', 'packet_log_word_144', 'packet_log_word_145', 'packet_log_word_146', 'packet_log_word_147', 'packet_log_word_148', 'packet_log_word_149', 'packet_log_word_150', 'packet_log_word_151', 'packet_log_word_152', 'packet_log_word_153', 'packet_log_word_154', 'packet_log_word_155', 'packet_log_word_156', 'packet_log_word_157', 'packet_log_word_158', 'packet_log_word_159', 'packet_log_word_160', 'packet_log_word_161', 'packet_log_word_162', 'packet_log_word_163', 'packet_log_word_164', 'packet_log_word_165', 'packet_log_word_166', 'packet_log_word_167', 'packet_log_word_168', 'packet_log_word_169', 'packet_log_word_170', 'packet_log_word_171', 'packet_log_word_172', 'packet_log_word_173', 'packet_log_word_174', 'packet_log_word_175', 'packet_log_word_176', 'packet_log_word_177', 'packet_log_word_178', 'packet_log_word_179', 'packet_log_word_180', 'packet_log_word_181', 'packet_log_word_182', 'packet_log_word_183', 'packet_log_word_184', 'packet_log_word_185', 'packet_log_word_186', 'packet_log_word_187', 'packet_log_word_188', 'packet_log_word_189', 'packet_log_word_190', 'packet_log_word_191', 'packet_log_word_192', 'packet_log_word_193', 'packet_log_word_194', 'packet_log_word_195', 'packet_log_word_196', 'packet_log_word_197', 'packet_log_word_198', 'packet_log_word_199', 'packet_log_word_200', 'packet_log_word_201', 'packet_log_word_202', 'packet_log_word_203', 'packet_log_word_204', 'packet_log_word_205', 'packet_log_word_206', 'packet_log_word_207', 'packet_log_word_208', 'packet_log_word_209', 'packet_log_word_210', 'packet_log_word_211', 'packet_log_word_212', 'packet_log_word_213', 'packet_log_word_214', 'packet_log_word_215', 'packet_log_word_216', 'packet_log_word_217', 'packet_log_word_218', 'packet_log_word_219', 'packet_log_word_220', 'packet_log_word_221', 'packet_log_word_222', 'packet_log_word_223', 'packet_log_word_224', 'packet_log_word_225', 'packet_log_word_226', 'packet_log_word_227', 'packet_log_word_228', 'packet_log_word_229', 'packet_log_word_230', 'packet_log_word_231', 'packet_log_word_232', 'packet_log_word_233', 'packet_log_word_234', 'packet_log_word_235', 'packet_log_word_236', 'packet_log_word_237', 'packet_log_word_238', 'packet_log_word_239', 'packet_log_word_240', 'packet_log_word_241', 'packet_log_word_242', 'packet_log_word_243', 'packet_log_word_244', 'packet_log_word_245', 'packet_log_word_246', 'packet_log_word_247', 'packet_log_word_248', 'packet_log_word_249', 'packet_log_word_250', 'packet_log_word_251', 'packet_log_word_252', 'packet_log_word_253', 'packet_log_word_254', 'packet_log_word_255', 'packet_log_word_256', 'packet_log_word_257', 'packet_log_word_258', 'packet_log_word_259', 'packet_log_word_260', 'packet_log_word_261', 'packet_log_word_262', 'packet_log_word_263', 'packet_log_word_264', 'packet_log_word_265', 'packet_log_word_266', 'packet_log_word_267', 'packet_log_word_268', 'packet_log_word_269', 'packet_log_word_270', 'packet_log_word_271', 'packet_log_word_272', 'packet_log_word_273', 'packet_log_word_274', 'packet_log_word_275', 'packet_log_word_276', 'packet_log_word_277', 'packet_log_word_278', 'packet_log_word_279', 'packet_log_word_280', 'packet_log_word_281', 'packet_log_word_282', 'packet_log_word_283', 'packet_log_word_284', 'packet_log_word_285', 'packet_log_word_286', 'packet_log_word_287', 'packet_log_word_288', 'packet_log_word_289', 'packet_log_word_290', 'packet_log_word_291', 'packet_log_word_292', 'packet_log_word_293', 'packet_log_word_294', 'packet_log_word_295', 'packet_log_word_296', 'packet_log_word_297', 'packet_log_word_298', 'packet_log_word_299', 'packet_log_word_300', 'packet_log_word_301', 'packet_log_word_302', 'packet_log_word_303', 'packet_log_word_304', 'packet_log_word_305', 'packet_log_word_306', 'packet_log_word_307', 'packet_log_word_308', 'packet_log_word_309', 'packet_log_word_310', 'packet_log_word_311', 'packet_log_word_312', 'packet_log_word_313', 'packet_log_word_314', 'packet_log_word_315', 'packet_log_word_316', 'packet_log_word_317', 'packet_log_word_318', 'packet_log_word_319', 'packet_log_word_320', 'packet_log_word_321', 'packet_log_word_322', 'packet_log_word_323', 'packet_log_word_324', 'packet_log_word_325', 'packet_log_word_326', 'packet_log_word_327', 'packet_log_word_328', 'packet_log_word_329', 'packet_log_word_330', 'packet_log_word_331', 'packet_log_word_332', 'packet_log_word_333', 'packet_log_word_334', 'packet_log_word_335', 'packet_log_word_336', 'packet_log_word_337', 'packet_log_word_338', 'packet_log_word_339', 'packet_log_word_340', 'packet_log_word_341', 'packet_log_word_342', 'packet_log_word_343', 'packet_log_word_344', 'packet_log_word_345', 'packet_log_word_346', 'packet_log_word_347', 'packet_log_word_348', 'packet_log_word_349', 'packet_log_word_350', 'packet_log_word_351', 'packet_log_word_352', 'packet_log_word_353', 'packet_log_word_354', 'packet_log_word_355', 'packet_log_word_356', 'packet_log_word_357', 'packet_log_word_358', 'packet_log_word_359', 'packet_log_word_360', 'packet_log_word_361', 'packet_log_word_362', 'packet_log_word_363', 'packet_log_word_364', 'packet_log_word_365', 'packet_log_word_366', 'packet_log_word_367', 'packet_log_word_368', 'packet_log_word_369', 'packet_log_word_370', 'packet_log_word_371', 'packet_log_word_372', 'packet_log_word_373', 'packet_log_word_374', 'packet_log_word_375', 'packet_log_word_376', 'packet_log_word_377', 'packet_log_word_378', 'packet_log_word_379', 'packet_log_word_380', 'packet_log_word_381', 'packet_log_word_382', 'packet_log_word_383', 'packet_log_word_384', 'packet_log_word_385', 'packet_log_word_386', 'packet_log_word_387', 'packet_log_word_388', 'packet_log_word_389', 'packet_log_word_390', 'packet_log_word_391', 'packet_log_word_392', 'packet_log_word_393', 'packet_log_word_394', 'packet_log_word_395', 'packet_log_word_396', 'packet_log_word_397', 'packet_log_word_398', 'packet_log_word_399', 'packet_log_word_400', 'packet_log_word_401', 'packet_log_word_402', 'packet_log_word_403', 'packet_log_word_404', 'packet_log_word_405', 'packet_log_word_406', 'packet_log_word_407', 'packet_log_word_408', 'packet_log_word_409', 'packet_log_word_410', 'packet_log_word_411', 'packet_log_word_412', 'packet_log_word_413', 'packet_log_word_414', 'packet_log_word_415', 'packet_log_word_416', 'packet_log_word_417', 'packet_log_word_418', 'packet_log_word_419', 'packet_log_word_420', 'packet_log_word_421', 'packet_log_word_422', 'packet_log_word_423', 'packet_log_word_424', 'packet_log_word_425', 'packet_log_word_426', 'packet_log_word_427', 'packet_log_word_428', 'packet_log_word_429', 'packet_log_word_430', 'packet_log_word_431', 'packet_log_word_432', 'packet_log_word_433', 'packet_log_word_434', 'packet_log_word_435', 'packet_log_word_436', 'packet_log_word_437', 'packet_log_word_438', 'packet_log_word_439', 'packet_log_word_440', 'packet_log_word_441', 'packet_log_word_442', 'packet_log_word_443', 'packet_log_word_444', 'packet_log_word_445', 'packet_log_word_446', 'packet_log_word_447', 'packet_log_word_448', 'packet_log_word_449', 'packet_log_word_450', 'packet_log_word_451', 'packet_log_word_452', 'packet_log_word_453', 'packet_log_word_454', 'packet_log_word_455', 'packet_log_word_456', 'packet_log_word_457', 'packet_log_word_458', 'packet_log_word_459', 'packet_log_word_460', 'packet_log_word_461', 'packet_log_word_462', 'packet_log_word_463', 'packet_log_word_464', 'packet_log_word_465', 'packet_log_word_466', 'packet_log_word_467', 'packet_log_word_468', 'packet_log_word_469', 'packet_log_word_470', 'packet_log_word_471', 'packet_log_word_472', 'packet_log_word_473', 'packet_log_word_474', 'packet_log_word_475', 'packet_log_word_476', 'packet_log_word_477', 'packet_log_word_478', 'packet_log_word_479', 'packet_log_word_480', 'packet_log_word_481', 'packet_log_word_482', 'packet_log_word_483', 'packet_log_word_484', 'packet_log_word_485', 'packet_log_word_486', 'packet_log_word_487', 'packet_log_word_488', 'packet_log_word_489', 'packet_log_word_490', 'packet_log_word_491', 'packet_log_word_492', 'packet_log_word_493', 'packet_log_word_494', 'packet_log_word_495', 'packet_log_word_496', 'packet_log_word_497', 'packet_log_word_498', 'packet_log_word_499', 'packet_log_word_500', 'packet_log_word_501', 'packet_log_word_502', 'packet_log_word_503', 'packet_log_word_504', 'packet_log_word_505', 'packet_log_word_506', 'packet_log_word_507', 'packet_log_word_508', 'packet_log_word_509', 'packet_log_word_510', 'packet_log_word_511')
WORD_NAMES = WORDS
STAGES = ('kernel', 'before_finalize', 'entry_config', 'init_prepare', 'magic', 'count_init', 'role_init', 'count_send', 'sender', 'count_receive', 'count_prefix', 'count_complete', 'count_drain', 'receiver', 'final_drain', 'count_ready_poll', 'cumsum_join', 'prefix_ready', 'recv_prefix_load', 'recv_tasks', 'recv_buffers', 'recv_packet_poll', 'recv_load_consume', 'recv_loop_clear', 'recv_final_drain', 'send_buffers_routes', 'send_packets', 'send_drain_metadata', 'recv_first_packet',
           'send_count_first_frame', 'send_count_last_frame', 'recv_count_arrive', 'recv_count_poll_iter')
STAGE_INFO = {
    'kernel': {'label': '核总耗时', 'start': 0, 'end': 21, 'kind': 'difference', 'boundary': '核入口至最终 PIPE_ALL/CTRL 恢复完成；包含区间内打点开销，不含末尾诊断身份/有效位发布及 cache 写回', 'relationship': '主阶段', 'reference_formula': '(w21-w0)/1000'},
    'before_finalize': {'label': '写出前总耗时', 'start': 0, 'end': 57, 'kind': 'difference', 'boundary': '核入口至诊断 Finish 开始；包含此前主点 scalar GM 写入，不含随后身份/有效位发布及 cache 写回', 'relationship': '总计', 'reference_formula': '(w57-w0)/1000'},
    'entry_config': {'label': '入口配置', 'start': 0, 'end': 1, 'kind': 'difference', 'boundary': '入口配置和 tracer 初始化', 'relationship': '主阶段', 'reference_formula': '(w1-w0)/1000'},
    'init_prepare': {'label': 'Init 准备', 'start': 1, 'end': 2, 'kind': 'difference', 'boundary': 'kernel Init 绑定至 magic 开始', 'relationship': '主阶段', 'reference_formula': '(w2-w1)/1000'},
    'magic': {'label': 'magic', 'start': 2, 'end': 3, 'kind': 'difference', 'boundary': '窗口 magic 轮转，沿用原 DMA/事件/TPipe Reset', 'relationship': '主阶段', 'reference_formula': '(w3-w2)/1000'},
    'count_init': {'label': '计数 Init', 'start': 3, 'end': 4, 'kind': 'difference', 'boundary': '计数参数、控制区、CTRL 配置', 'relationship': '主阶段', 'reference_formula': '(w4-w3)/1000'},
    'role_init': {'label': '收发 Init', 'start': 4, 'end': 5, 'kind': 'difference', 'boundary': 'sender/receiver Init 和布局计算', 'relationship': '主阶段', 'reference_formula': '(w5-w4)/1000'},
    'count_send': {'label': '计数统计/发送提交', 'start': 6, 'end': 7, 'kind': 'difference', 'boundary': 'SendCnt，包含已有同步，远端写可异步继续', 'relationship': '主阶段', 'reference_formula': '(w7-w6)/1000'},
    'sender': {'label': '发送 payload/元数据完成', 'start': 8, 'end': 9, 'kind': 'difference', 'boundary': 'Sender.Process，结束已有 MTE3_S 等待', 'relationship': '发送主阶段', 'reference_formula': '(w9-w8)/1000'},
    'count_receive': {'label': '计数接收/归约', 'start': 10, 'end': 11, 'kind': 'difference', 'boundary': 'RecvCnt，包含 ready 轮询及归约；普通路径在此清计数标记，NDR packed-count 路径随后在 CalcCumsum 中清理', 'relationship': '接收主阶段', 'reference_formula': '(w11-w10)/1000'},
    'count_prefix': {'label': 'cumsum 汇合/发布提交', 'start': 12, 'end': 13, 'kind': 'difference', 'boundary': 'CalcCumsum，包含 GetCumSum 和前缀发布；NDR packed-count 清计数标记位于 GetCumSum 返回后，不计入其子段', 'relationship': '接收主阶段', 'reference_formula': '(w13-w12)/1000'},
    'count_complete': {'label': '前缀就绪/加载/输出提交', 'start': 14, 'end': 15, 'kind': 'difference', 'boundary': 'CompleteCumsum，输出计数仍可异步继续', 'relationship': '接收主阶段', 'reference_formula': '(w15-w14)/1000'},
    'count_drain': {'label': '计数阶段 drain', 'start': 16, 'end': 17, 'kind': 'difference', 'boundary': '现有 PIPE_ALL，回收计数 UB 前完成写入', 'relationship': '接收主阶段', 'reference_formula': '(w17-w16)/1000'},
    'receiver': {'label': '接收 payload 完成', 'start': 18, 'end': 19, 'kind': 'difference', 'boundary': 'Receiver.Process，含构建任务、轮询、接收及收尾', 'relationship': '接收主阶段', 'reference_formula': '(w19-w18)/1000'},
    'final_drain': {'label': '最终 drain/恢复 CTRL', 'start': 20, 'end': 21, 'kind': 'difference', 'boundary': 'Counts.Finish，包含现有 PIPE_ALL', 'relationship': '主阶段', 'reference_formula': '(w21-w20)/1000'},
    'count_ready_poll': {'label': '计数 ready 轮询', 'start': 24, 'end': 25, 'kind': 'difference', 'boundary': 'WaitDispatch 轮询至最后一次 V_S 完成', 'relationship': '计数接收/归约的细分', 'reference_formula': '(w25-w24)/1000'},
    'cumsum_join': {'label': 'cumsum 跨核等待/清理提交', 'start': 26, 'end': 27, 'kind': 'difference', 'boundary': 'GetCumSum，包含等待片段总数；子槽回收在后续前缀屏障后执行', 'relationship': 'cumsum 汇合/发布提交的细分', 'reference_formula': '(w27-w26)/1000'},
    'prefix_ready': {'label': '前缀 ready 等待/清理提交', 'start': 28, 'end': 29, 'kind': 'difference', 'boundary': 'WaitCumSumFlag 等待及清理提交', 'relationship': '前缀就绪/加载/输出提交的细分', 'reference_formula': '(w29-w28)/1000'},
    'recv_prefix_load': {'label': '接收前缀加载', 'start': 18, 'end': 30, 'kind': 'difference', 'boundary': 'Receiver reset / 前缀 DMA，结束 MTE2_S', 'relationship': '接收 payload 完成的细分', 'reference_formula': '(w30-w18)/1000'},
    'recv_tasks': {'label': '接收任务构建', 'start': 30, 'end': 31, 'kind': 'difference', 'boundary': 'ReceiveTasks.Build', 'relationship': '接收 payload 完成的细分', 'reference_formula': '(w31-w30)/1000'},
    'recv_buffers': {'label': '接收缓冲准备', 'start': 31, 'end': 32, 'kind': 'difference', 'boundary': '探测、lane、标记缓冲和事件准备', 'relationship': '接收 payload 完成的细分', 'reference_formula': '(w32-w31)/1000'},
    'recv_packet_poll': {'label': 'packet 就绪轮询累计', 'start': None, 'end': 33, 'kind': 'sum', 'boundary': '每批完整轮询区间累积，未对每次空扫描插点', 'relationship': '接收 payload 完成的细分', 'reference_formula': 'w33/1000'},
    'recv_load_consume': {'label': '加载/消费提交累计', 'start': None, 'end': 34, 'kind': 'sum', 'boundary': 'IssueLoad / ConsumeLoaded / RemovePending 区间累积', 'relationship': '接收 payload 完成的细分', 'reference_formula': 'w34/1000'},
    'recv_loop_clear': {'label': '接收循环/清标记提交', 'start': 32, 'end': 35, 'kind': 'difference', 'boundary': '包含 packet 循环、lane 等待指令和 ClearFlags 提交', 'relationship': '接收 payload 完成的细分', 'reference_formula': '(w35-w32)/1000'},
    'recv_final_drain': {'label': '接收最终 PIPE_ALL', 'start': 35, 'end': 36, 'kind': 'difference', 'boundary': '接收输出、转发及标记清理的现有完成屏障', 'relationship': '接收 payload 完成的细分', 'reference_formula': '(w36-w35)/1000'},
    'send_buffers_routes': {'label': '发送缓冲/路由加载', 'start': 8, 'end': 38, 'kind': 'difference', 'boundary': 'Sender UB/lane 分配及路由加载，含已有同步', 'relationship': '发送 payload/元数据完成的细分', 'reference_formula': '(w38-w8)/1000'},
    'send_packets': {'label': '路由/packet 发送提交', 'start': 38, 'end': 39, 'kind': 'difference', 'boundary': 'SendWithDedup / SendWithoutDedup，包含已有 lane 依赖', 'relationship': '发送 payload/元数据完成的细分', 'reference_formula': '(w39-w38)/1000'},
    'send_drain_metadata': {'label': '发送 drain/元数据完成', 'start': 39, 'end': 40, 'kind': 'difference', 'boundary': '尾 RDMA batch Flush（消息完成quiet）、lane 回收、source metadata 写回及已有 MTE3_S；无尾批次时不执行 Flush', 'relationship': '发送 payload/元数据完成的细分', 'reference_formula': '(w40-w39)/1000'},
    'recv_first_packet': {'label': '首次 packet 就绪等待', 'start': 32, 'end': 42, 'kind': 'difference', 'boundary': '接收缓冲准备完至首个 packet batch ready', 'relationship': '接收 payload 完成的细分', 'reference_formula': '(w42-w32)/1000'},
    'send_count_first_frame': {'label': '远端计数首帧下发', 'start': 6, 'end': 44, 'kind': 'difference', 'boundary': 'SendCnt 开始至本核首个远端 count 帧提交完成', 'relationship': '计数统计/发送提交的细分', 'reference_formula': '(w44-w6)/1000'},
    'send_count_last_frame': {'label': '远端计数逐帧下发跨度', 'start': 44, 'end': 45, 'kind': 'difference', 'boundary': '本核首个远端 count 帧提交完成至本核末个远端帧提交完成', 'relationship': '计数统计/发送提交的细分', 'reference_formula': '(w45-w44)/1000'},
    'recv_count_arrive': {'label': '计数 ready 首次到达', 'start': 10, 'end': 46, 'kind': 'difference', 'boundary': 'RecvCnt 开始至首次观测到非零 ready 汇总', 'relationship': '计数接收/归约的细分', 'reference_formula': '(w46-w10)/1000'},
    'recv_count_poll_iter': {'label': '首轮 ready 轮询', 'start': 24, 'end': 47, 'kind': 'difference', 'boundary': 'ready 轮询开始至首轮 MTE2/V_S 扫描完成', 'relationship': '计数 ready 轮询的细分', 'reference_formula': '(w47-w24)/1000'},
}

DATA_WORDS = frozenset((*range(22), *range(24, 50), 57, 62))
COUNT_WORDS = frozenset((33, 34, 37, 41, 43, 48, 49, 62))
VALID_BITS = sum(1 << index for index in DATA_WORDS)
RESERVED_WORDS = frozenset((22, 23, 63))
COMMON_REQUIRED = frozenset((*range(8), 20, 21, 57))
SEND_REQUIRED = frozenset((8, 9, 43))
RECEIVE_REQUIRED = frozenset((*range(10, 20), 30, 31, 33, 34, 37, 41))


SENDER_PACKET_METRICS = {
    'second_schedule_without_q': '第二跨机包调度（扣q和前包提交）',
    'second_previous_submit': '第二包准备后提交前包的时间',
    'second_defer_gap': '第二包暂存提交至开始等完成',
    'second_stage_wait': '第二包暂存完成剩余等待',
    'second_member_q': '第二跨机包成员q区间',
    'second_stage': '第二跨机包暂存至put前经过（含重叠）',
    'second_put_call': '第二跨机包put调用',
    'second_enqueue_to_put_return': '第二跨机包Enqueue至put返回',
    'remote_submit_span': '跨机包首开始至末put返回',
    'remote_retire_span': '跨机包首开始至末批quiet返回',
    'group_submit_span': '本组包首开始至末提交返回',
    'batch_quiet_total': '全部批次quiet区间累计',
}


def parse_packet_detail(words, role, frequency_hz):
    if words[64] != 1 or any(words[108:128]):
        raise ValueError('Invalid packet detail metadata')
    def difference(a, b):
        if not words[0] <= words[a] <= words[b] <= words[57]:
            raise ValueError('Invalid packet interval')
        return words[b] - words[a]
    if role == 'send':
        if any(words[96:]):
            raise ValueError('Sender has receiver packet records')
        remote, group = words[65], words[66]
        if remote + group > words[88] or words[89] > 8 or words[93] > remote or (words[89] < 2 and words[93]):
            raise ValueError('Invalid sender packet counters')
        ticks = {}
        if remote:
            ticks['remote_submit_span'] = difference(67, 68)
            ticks['remote_retire_span'] = difference(67, 82)
            ticks['batch_quiet_total'] = words[80]
            if not 1 <= words[81] <= remote or not 1 <= words[84] <= words[85] <= words[88]:
                raise ValueError('Invalid remote batch/order counters')
            if words[80] > words[21] - words[0] or words[82] < words[68]:
                raise ValueError('Invalid remote retirement')
        elif any(words[i] for i in (67,68,80,81,82,84,85)):
            raise ValueError('Empty remote path has observations')
        if group:
            ticks['group_submit_span'] = difference(69, 70)
            if not 1 <= words[86] <= words[87] <= words[88]:
                raise ValueError('Invalid group packet order')
        elif any(words[i] for i in (69,70,86,87)):
            raise ValueError('Empty group path has observations')
        if remote >= 2:
            schedule = difference(71, 72)
            if words[75]+words[94] > schedule or words[83] not in (0,1) or words[95] not in (0,1):
                raise ValueError('Invalid second-packet q/flush accounting')
            difference(72,90); difference(92,73)
            ticks.update(second_schedule_without_q=schedule-words[75]-words[94], second_member_q=words[75],
                         second_previous_submit=words[94], second_defer_gap=difference(90,91), second_stage_wait=difference(91,92),
                         second_stage=difference(72,73), second_put_call=difference(73,74),
                         second_enqueue_to_put_return=difference(72,74))
            second = dict(token=words[76], header_k=words[77], expert=words[78], peer=words[79],
                          triggered_flush=bool(words[83]), deferred=bool(words[95]))
        else:
            if any(words[71:80]) or words[83] or any(words[90:93]) or words[94] or words[95]:
                raise ValueError('Absent second packet has detail')
            second = None
        return dict(kind='sender', remote_packets=remote, group_packets=group, packet_count=words[88],
                    lanes=words[89], second=second, batch_flushes=words[81], deferred_packets=words[93], ticks=ticks,
                    durations={k: v*1e6/frequency_hz if frequency_hz is not None else v for k,v in ticks.items()},
                    packet_order_interleaves=bool(remote and group and max(words[84],words[86]) <= min(words[85],words[87])),
                    intervals_interleave=bool(remote and group and max(words[67],words[69]) <= min(words[68],words[70])),
                    raw_metadata=words[64:128])
    if any(words[65:96]):
        raise ValueError('Receiver has sender packet metadata')
    rows, capacity, total, ready, processed, lanes = words[96:102]
    if not 0 <= rows == capacity <= min(total,128) or total != words[37] or ready != total or processed != total:
        raise ValueError('Invalid receive trace coverage')
    if not 0 <= lanes <= 8 or words[107] != int(rows < total):
        raise ValueError('Invalid receive trace lanes/truncation')
    export_ticks = difference(102,103)
    if words[102] < words[21]:
        raise ValueError('Receive trace exported before business completion')
    if words[105] != words[54] or words[106] < 1:
        raise ValueError('Invalid receive trace shape')
    entries=[]
    for j in range(rows):
        packed=words[128+j*4:132+j*4]
        fields=[x for value in packed for x in (value & 0xffffffff,value >> 32)]
        found,batch,order,source,token,k,role_members,slot=fields
        packet_role=role_members & 255; members=role_members >> 8
        if (not 1 <= found <= total or not 1 <= order <= total or not 1 <= batch <= words[41]
                or packet_role not in (1,2,3) or source >= words[52] or token >= words[54]
                or k >= 32 or members >= 32 or (packet_role != 2 and members)):
            raise ValueError('Invalid receive packet identity/order')
        local_expert=(slot // words[105]) % words[106]
        if ((slot // words[105]) // words[106] - words[50]) % words[52] != source:
            raise ValueError('Receive slot/source mismatch')
        # 任务按固定步长交错分配，输出行 = 首行 + task * 接收核数。
        recv_cores = words[53] // 2
        entries.append(dict(task=j, output_row=words[104] + j * recv_cores, ready_order=found, ready_batch=batch,
                            process_order=order, source_rank=source, source_token=token, header_k=k,
                            packet_role={1:'DIRECT',2:'FIRST_HOP',3:'TERMINAL'}[packet_role],
                            forward_members=members, destination_expert=words[50]*words[106]+local_expert,
                            q=slot % words[105], slot=slot))
    for field in ('ready_order','process_order'):
        if len({e[field] for e in entries}) != rows:
            raise ValueError('Duplicate receive trace order')
    if any(words[128+rows*4:]):
        raise ValueError('Nonzero unused receive trace storage')
    return dict(kind='receiver', entries=entries, total_tasks=total, recorded_tasks=rows,
                truncated=bool(words[107]), lanes=lanes, export_ticks=export_ticks,
                raw_metadata=words[64:128])


def parse_records(raw, operation='dispatch', frequency_hz=DEFAULT_FREQUENCY_HZ):
    """Validate a complete capture and return 29 reference-compatible durations."""
    if operation != 'dispatch':
        raise ValueError('Reference profile supports dispatch only')
    if frequency_hz is not None and (isinstance(frequency_hz, bool) or
            not isinstance(frequency_hz, (int, float)) or
            not math.isfinite(frequency_hz) or frequency_hz <= 0):
        raise ValueError('Counter frequency must be finite and positive')
    if not isinstance(raw, (list, tuple)) or not 16 <= len(raw) <= 72 or len(raw) % 2:
        raise ValueError('Expected actual even AIV count in [16,72]')
    records, cores, identity = [], [], None
    for core, slots in enumerate(raw):
        if len(slots) != SLOTS or any(len(row) != FIELDS for row in slots):
            raise ValueError('Reference timer shape must be [active_aiv,80,8]')
        words = [value for row in slots for value in row]
        if any(type(value) is not int or not 0 <= value < 2**64 for value in words):
            raise ValueError('Invalid uint64 raw word')
        if any(words[index] for index in RESERVED_WORDS):
            raise ValueError('Nonzero reserved word')
        if words[51] != core or words[53] != len(raw):
            raise ValueError('Core identity or active AIV count mismatch')
        if words[55:57] != [SCHEMA, 1] or words[59:62] != [MAGIC, 1, 1]:
            raise ValueError('Incomplete or mismatched reference header')
        rank, world, tokens = words[50], words[52], words[54]
        if world not in (16, 32, 64, 128, 256) or rank >= world or not 1 <= tokens <= 65536:
            raise ValueError('Invalid rank/world identity')
        current = (rank, world, tokens)
        if identity is not None and current != identity:
            raise ValueError('Mixed rank/world/BS identity across cores')
        identity = current
        mask = words[58]
        if mask & ~VALID_BITS:
            raise ValueError('Validity mask marks metadata or reserved words')
        valid = {index for index in DATA_WORDS if mask & (1 << index)}
        if not COMMON_REQUIRED.issubset(valid):
            raise ValueError('Missing mandatory common reference points')
        if any(words[index] for index in DATA_WORDS - valid):
            raise ValueError('Unmarked data word must remain zero')
        if not words[0] <= words[21] <= words[57]:
            raise ValueError('Invalid completed kernel timestamp order')
        for index in valid - COUNT_WORDS:
            if not words[0] <= words[index] <= words[57]:
                raise ValueError('Timestamp outside own core interval')
        role = 'send' if core < len(raw)//2 else 'receive'
        required = SEND_REQUIRED if role == 'send' else RECEIVE_REQUIRED
        if not required.issubset(valid):
            raise ValueError('Missing mandatory ' + role + ' reference points')
        task_word = 43 if role == 'send' else 37
        if role == 'send' and valid.intersection({*range(10, 20), *range(24, 38), 41, 42, 46, 47, 48, 49}):
            raise ValueError('Send core contains receive-only points')
        if role == 'receive' and valid.intersection({8, 9, 38, 39, 40, 43, 44, 45}):
            raise ValueError('Receive core contains send-only points')
        task_points = {38, 39, 40} if role == 'send' else {32, 35, 36, 42}
        if words[task_word] == 0:
            if valid.intersection(task_points):
                raise ValueError('Empty ' + role + ' core contains task-only timestamps')
            if role == 'receive' and (words[33] or words[34] or words[41] or words[48] or words[49]):
                raise ValueError('Empty receive core has nonzero accumulation or packet batches')
        else:
            if not task_points.issubset(valid):
                raise ValueError('Nonempty ' + role + ' core lacks task timestamps')
            if role == 'receive' and not 1 <= words[41] <= words[37]:
                raise ValueError('Receive packet batches must be in [1, task_count]')
        # Count ownership and payload task ownership are different partitions.
        # Count detail pairs may be absent even on a nonempty receiver; never
        # infer their execution from w37. A single endpoint is always corrupt.
        for first, last in ((24, 25), (26, 27), (28, 29)):
            if (first in valid) != (last in valid):
                raise ValueError('Incomplete optional count interval')
        for stage in STAGES:
            info = STAGE_INFO[stage]
            start, end = info['start'], info['end']
            if end not in valid or (start is not None and start not in valid):
                continue
            first = words[start] if start is not None else None
            last = words[end] if start is not None else None
            total = words[end] - first if start is not None else words[end]
            if total < 0:
                raise ValueError('Invalid timestamp order for ' + stage)
            if start is None and total > words[21] - words[0]:
                raise ValueError('Accumulated interval exceeds own kernel')
            calls = words[41] if start is None else 1
            record = dict(core=core, role=role, stage=stage, first_tick=first,
                          last_tick=last, total_ticks=total, calls=calls)
            if frequency_hz is not None:
                record['total_us'] = total * 1e6 / frequency_hz
            records.append(record)
        cores.append(dict(core=core, role=role, task_count=words[task_word],
                          batch_count=words[41] if role == 'receive' else None,
                          valid_mask=mask, packet_detail=parse_packet_detail(words, role, frequency_hz)))
    return dict(format=FORMAT, schema=SCHEMA, operation='dispatch', active_aiv=len(raw),
                boundary_mode='reference', frequency_hz=frequency_hz, rank=identity[0],
                world_size=identity[1], num_tokens=identity[2], records=records, cores=cores,
                clock='AscendC::GetSystemCycle; independent core clocks',
                interpretation='Same-core integer differences; nested stages cannot be added; '
                               'w33/w34 are accumulated cycles, not first-to-last envelopes.')
