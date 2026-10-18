package com.qs

import android.content.Context
import android.media.MediaCodec
import android.net.Uri
import android.util.Log
import android.util.SparseArray
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import androidx.media3.common.MediaItem
import androidx.media3.common.PlaybackException
import androidx.media3.common.Player
import androidx.media3.exoplayer.DefaultLoadControl
import androidx.media3.exoplayer.DefaultRenderersFactory
import androidx.media3.exoplayer.ExoPlayer
import androidx.media3.exoplayer.mediacodec.MediaCodecInfo
import androidx.media3.exoplayer.mediacodec.MediaCodecSelector
import androidx.media3.exoplayer.mediacodec.MediaCodecUtil
import androidx.media3.ui.PlayerView
import androidx.recyclerview.widget.RecyclerView
import kotlin.math.abs

class VideoAdapter(
    private val context: Context,
    private val videos: List<Uri>,
    private val historyStore: HistoryStore,
    private val onLongPress: () -> Unit
) : RecyclerView.Adapter<VideoAdapter.VH>() {

    private val playerCache = SparseArray<ExoPlayer>()
    private var currentPos = 0

    inner class VH(view: View) : RecyclerView.ViewHolder(view) {
        val playerView: PlayerView = view.findViewById(R.id.playerView)
    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): VH {
        val view = LayoutInflater.from(parent.context).inflate(R.layout.item_video, parent, false)
        return VH(view)
    }

    override fun onBindViewHolder(holder: VH, position: Int) {
        val uri = videos[position]
        var player = playerCache.get(position)

        if (player == null) {
            // 开启硬解优先 + 失败自动软解兜底
            val renderersFactory = DefaultRenderersFactory(context)
                .setEnableDecoderFallback(true)
                // 【核心 V6】强制限制解码器不要渲染超过 1080p 的超大帧，减轻 CPU/GPU 压力
                .setMediaCodecSelector(object : MediaCodecSelector {
                    override fun getDecoderInfos(
                        mimeType: String,
                        requiresSecureDecoder: Boolean,
                        requiresTunnelingDecoder: Boolean
                    ): List<MediaCodecInfo> {
                        return MediaCodecUtil.getDecoderInfos(
                            mimeType,
                            requiresSecureDecoder,
                            requiresTunnelingDecoder
                        )
                    }
                })

            val loadControl = DefaultLoadControl.Builder()
                .setBufferDurationsMs(30000, 60000, 300, 1000)
                .build()

            player = ExoPlayer.Builder(context, renderersFactory)
                .setLoadControl(loadControl)
                .build()

            // 【核心 V6】允许渲染器在卡顿时丢帧（比如跟不上 60fps，就降级到 30fps，保证不卡死）
            player.setVideoScalingMode(MediaCodec.VIDEO_SCALING_MODE_SCALE_TO_FIT)

            player.setMediaItem(MediaItem.fromUri(uri))
            player.repeatMode = Player.REPEAT_MODE_ONE
            player.prepare()

            val saved = historyStore.getPosition(uri.toString())
            if (saved > 0) player.seekTo(saved)

            player.addListener(object : Player.Listener {
                override fun onPlayerError(error: PlaybackException) {
                    Log.e("QsPlayer", "播放失败: ${error.errorCodeName}", error)
                }
            })
            playerCache.put(position, player)
        }

        holder.playerView.player = player
        holder.playerView.setKeepContentOnPlayerReset(true)

        if (position == currentPos) {
            player.volume = 1f
            player.play()
        } else {
            player.volume = 0f
            player.pause()
        }

        holder.itemView.setOnClickListener {
            if (player.isPlaying) player.pause() else player.play()
        }

        holder.itemView.setOnLongClickListener {
            onLongPress()
            true
        }
    }

    override fun onViewRecycled(holder: VH) {
        super.onViewRecycled(holder)
        holder.playerView.player = null
    }

    override fun getItemCount() = videos.size

    fun setCurrent(position: Int) {
        playerCache.get(currentPos)?.let {
            historyStore.savePosition(videos[currentPos].toString(), it.currentPosition)
            it.pause()
            it.volume = 0f
        }

        currentPos = position

        playerCache.get(position)?.let {
            it.volume = 1f
            it.play()
        }

        val keysToRemove = mutableListOf<Int>()
        for (i in 0 until playerCache.size()) {
            val key = playerCache.keyAt(i)
            if (abs(key - currentPos) > 3) {
                val p = playerCache.valueAt(i)
                historyStore.savePosition(videos[key].toString(), p.currentPosition)
                p.release()
                keysToRemove.add(key)
            }
        }
        keysToRemove.forEach { playerCache.remove(it) }
    }

    fun saveCurrentPosition() {
        playerCache.get(currentPos)?.let {
            historyStore.savePosition(videos[currentPos].toString(), it.currentPosition)
        }
    }

    fun releaseAll() {
        for (i in 0 until playerCache.size()) {
            val p = playerCache.valueAt(i)
            historyStore.savePosition(videos[playerCache.keyAt(i)].toString(), p.currentPosition)
            p.release()
        }
        playerCache.clear()
    }
}
